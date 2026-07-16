import numba
import numpy as np
from numba import jit, njit
import pandas as pd


@jit(nopython=True)
def is_inside_sm(polygon, point):
    length = len(polygon) - 1
    dy2 = point[1] - polygon[0][1]
    intersections = 0
    ii = 0
    jj = 1

    while ii < length:
        dy = dy2
        dy2 = point[1] - polygon[jj][1]

        # consider only lines which are not completely above/bellow/right from the point
        if dy * dy2 <= 0.0 and (point[0] >= polygon[ii][0] or point[0] >= polygon[jj][0]):

            # non-horizontal line
            if dy < 0 or dy2 < 0:
                F = dy * (polygon[jj][0] - polygon[ii][0]) / (dy - dy2) + polygon[ii][0]

                if point[0] > F:  # if line is left from the point - the ray moving towards left, will intersect it
                    intersections += 1
                elif point[0] == F:  # point on line
                    return 2

            # point on upper peak (dy2=dx2=0) or horizontal line (dy=dy2=0 and dx*dx2<=0)
            elif dy2 == 0 and (point[0] == polygon[jj][0] or (
                    dy == 0 and (point[0] - polygon[ii][0]) * (point[0] - polygon[jj][0]) <= 0)):
                return 2

        ii = jj
        jj += 1

    print('intersections =', intersections)
    return intersections & 1


@njit(parallel=True)
def is_inside_sm_parallel(points, polygon):
    ln_ = len(points)
    D = np.empty(ln_, dtype=numba.boolean)
    for i in numba.prange(ln_):
        D[i] = is_inside_sm(polygon, points[i])
    return D


# read geometry
cav_geom = pd.read_csv(r'D:\Dropbox\PyMultipact\C3795\geodata.n', header=None,
                       sep='\s+', engine='python')[[1, 0]]
print(len(cav_geom))
# the representation of a point will be a tuple (x,y)
# the representation of a polygon wil be a list of points [(x1,y1), (x2,y2), (x3,y3), ... ]

import sys
import copy
import time
import random
import matplotlib.pyplot as plt


def main(cav_geom):
    # -------------------------------------------------------------------------------
    scatter_points = []

    # random.seed(1389)
    i = 0
    t = time.time()
    while i < 5:
        testpoint = (random.randrange(-100, 100)*1e-3, random.randrange(0, 185)*1e-3)
        scatter_points.append(testpoint)
        is_inside_sm(cav_geom, testpoint)
        i += 1

    print("is_inside_sm() - execution time: ", time.time() - t)

    # start = time.time()
    # xxx = is_inside_sm_parallel(scatter_points, cav_geom)
    # print('time: ', time.time() - start)

    # df = pd.DataFrame(scatter_points)
    # df['in_domain'] = xxx
    # print(df)

    return scatter_points


if __name__ == '__main__':

    # close path
    cav_geom = np.append(cav_geom.to_numpy(), [cav_geom.to_numpy()[0]], axis=0)
    plt.plot(cav_geom[:, 0], cav_geom[:, 1], 'r', lw=2, zorder=60000)

    scatter_points = main(cav_geom)

    plt.scatter(np.array(scatter_points)[:, 0], np.array(scatter_points)[:, 1], fc='None', ec='k')
    plt.show()
