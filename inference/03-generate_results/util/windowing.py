from numpy import outer, hanning #, linspace, cos, pi

def hanning_2d(N=128):
    """
    Generate 2D Hanning window coefficients.

    Note 1: numerical precision seems to sacrifice perfect symmetry.
    Note 2: complementary values do not add up to exactly 1.
    If either of those two properties is essential for your application, do not use this function.
    Function was updated, it's unclear if this is still the case
    """

    # # Generate indices
    # x = linspace(0, N-1, N) + 0.5
    # # Hanning window in 1D
    # h1d = (1 - cos(2 * pi * x / N)) / 2

    h1d = hanning(N) # numpy 1D implementation

    # Outer product for Hanning window in 2d
    h2d = outer(h1d, h1d)

    return h2d


if __name__ == '__main__':
    print(hanning_2d(6))
