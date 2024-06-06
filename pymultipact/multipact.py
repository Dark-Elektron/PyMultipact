# %%time
# # check with defined functions
#
# %matplotlib
# notebook
# # calling it a second time may prevent some graphics errors
# %matplotlib
# notebook
# plt.close()
# plt.rcParams["figure.figsize"] = (9, 4)
# import copy
# import time
# import matplotlib
# from matplotlib import cm
# import numba
# from scipy.interpolate import CubicSpline


class Multipact:
    def __init__(self, domain=None):
        self.domain = domain

    def set_domain(self, domain):
        self.domain = domain


if __name__ == '__main__':
    print()
