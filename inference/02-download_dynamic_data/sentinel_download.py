"""
Medusa ToolBox.
Copyright (C) 2017 ONERA, Alexandre Boulch
This program is free software; you can redistribute it
and/or modify it under the terms of the GNU General
Public License as published by the Free Software Foundation;
either version 3 of the License, or any later version.
This program is distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public
License along with this program; if not, write to the Free
Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
Boston, MA 02110-1301  USA
"""

# Adapted by R. C. Daudt

from datetime import date
import argparse
import json
from sentinelsat.sentinel import SentinelAPI, read_geojson, geojson_to_wkt
from getpass import getpass

parser = argparse.ArgumentParser(description='PyTorch sentinel crop')
parser.add_argument('--sentinel', type=int, default=2, metavar='N',
                    help='1 for sentinel1 and 2 for sentinel 2')
parser.add_argument('--data', type=str, default="../assets/data.json", metavar='N',
                    help='datafile')
parser.add_argument('--geojson', type=str, default="../assets/switzerland_blob.geojson", metavar='N', help="footprint")
parser.add_argument('--outdir', type=str, default="./downloaded/", metavar='N', help="Download folder")
args = parser.parse_args()

print("loading id...")
data = json.load(open(args.data))

startdate = date(data["startdate"][0], data["startdate"][1], data["startdate"][2])
enddate = date(data["enddate"][0], data["enddate"][1], data["enddate"][2])


print("connecting to sentinel API...")
# api = SentinelAPI(data["login"], getpass(prompt='Copernicus password for user {}: '.format(data["login"])), 'https://scihub.copernicus.eu/dhus')
if data["password"] == "":
    api = SentinelAPI(data["login"], getpass(prompt='Copernicus password for user {}: '.format(data["login"])), api_url='https://apihub.copernicus.eu/apihub/')
else:
    api = SentinelAPI(data["login"], data["password"], api_url='https://apihub.copernicus.eu/apihub/')


# search by polygon, time, and SciHub query keywords
# Docs: https://scihub.copernicus.eu/userguide/FullTextSearch
print("searching...")
footprint = geojson_to_wkt(read_geojson(args.geojson))
if args.sentinel == 1:
    products = api.query(footprint,
                         date=(startdate,enddate),
                         platformname = 'Sentinel-1',
                         producttype = "GRD"
                         )
elif args.sentinel == 2:
    products = api.query(footprint,
                         date=(startdate,enddate),
                         platformname = 'Sentinel-2',
                         cloudcoverpercentage=(0, 100),
                         producttype = 'S2MSI2A'
                         )
elif args.sentinel == 3:
    products = api.query(footprint,
                         date=(startdate,enddate),
                         platformname = 'Sentinel-3',
                         producttype = 'SL_2_LST___', #'SR_2_LAN___',
                         timeliness= 'Near Real Time'
#                          timeliness= 'Non Time Critical'
                         )
print("  product number: ",len(products))
# download all results from the search
print("downloading...")
api.download_all(products, directory_path=args.outdir)
