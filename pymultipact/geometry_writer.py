import matplotlib.pyplot as plt
import numpy as np
from icecream import ic
from scipy.optimize import fsolve


def write_ell_cavity(folder=None, mid_cell=None, lend_cell=None, rend_cell=None, beampipe=None, name=None,
                     step=None, n_cell=None, plot=False):

    """
    Write cavity geometry to be used for multipacting analysis

    Parameters
    ----------
    folder: str
        Folder path to write geometry to
    n_cell: int
        Number of cavity cells
    mid_cell: list, ndarray
        Array of cavity middle cells' geometric parameters
    lend_cell: list, ndarray
        Array of cavity left end cell's geometric parameters
    rend_cell: list, ndarray
        Array of cavity left end cell's geometric parameters
    beampipe: str {"left", "right", "both", "none"}
        Specify if beam pipe is on one or both ends or at no end at all
    plot: bool
        If True, the cavity geometry is plotted for viewing

    Returns
    -------

    """

    if mid_cell is None and lend_cell is None and rend_cell is None:
        A_m, B_m, a_m, b_m, Ri_m, L_m, Req_m = np.array([42, 42, 12, 19, 35, 57.6524, 103.353])*1e-3
        A_el, B_el, a_el, b_el, Ri_el, L_el, Req_el = np.array([42, 42, 12, 19, 35, 57.6524, 103.353])*1e-3
        A_er, B_er, a_er, b_er, Ri_er, L_er, Req_er = np.array([42, 42, 12, 19, 35, 57.6524, 103.353])*1e-3
    else:
        if lend_cell is None and rend_cell is None:
            A_m, B_m, a_m, b_m, Ri_m, L_m, Req_m = np.array(mid_cell)
            A_el, B_el, a_el, b_el, Ri_el, L_el, Req_el = np.array(mid_cell)
            A_er, B_er, a_er, b_er, Ri_er, L_er, Req_er = np.array(mid_cell)
        elif mid_cell is None and lend_cell is None:
            A_m, B_m, a_m, b_m, Ri_m, L_m, Req_m = np.array(rend_cell)
            A_el, B_el, a_el, b_el, Ri_el, L_el, Req_el = np.array(rend_cell)
            A_er, B_er, a_er, b_er, Ri_er, L_er, Req_er = np.array(rend_cell)
        elif mid_cell is None and rend_cell is None:
            A_m, B_m, a_m, b_m, Ri_m, L_m, Req_m = np.array(lend_cell)
            A_el, B_el, a_el, b_el, Ri_el, L_el, Req_el = np.array(lend_cell)
            A_er, B_er, a_er, b_er, Ri_er, L_er, Req_er = np.array(lend_cell)
        else:
            print("There is something wrong with the geometry definition. Reverts to the default TESLA geometry.")
            A_m, B_m, a_m, b_m, Ri_m, L_m, Req_m = np.array([42, 42, 12, 19, 35, 57.6524, 103.353])*1e-3
            A_el, B_el, a_el, b_el, Ri_el, L_el, Req_el = np.array([42, 42, 12, 19, 35, 57.6524, 103.353])*1e-3
            A_er, B_er, a_er, b_er, Ri_er, L_er, Req_er = np.array([42, 42, 12, 19, 35, 57.6524, 103.353])*1e-3
     
    if n_cell is None:       
        n_cell = 1
    if step is None:
        step = 0.005*min(L_m, L_el, L_er)  # step in boundary points in mm

    if beampipe is None or beampipe == 'None':
        L_bp_l = 0.000
        L_bp_r = 0.000
    elif beampipe.lower() == 'left':
        L_bp_l = 4*L_m
        L_bp_r = 0.000
    elif beampipe.lower() == 'right':
        L_bp_l = 0.000
        L_bp_r = 4*L_m
    elif beampipe.lower() == 'both':
        L_bp_l = 4*L_m
        L_bp_r = 4*L_m
    else:
        L_bp_l = 0.000
        L_bp_r = 0.000

    # calculate shift
    shift = (L_bp_r + L_bp_l + L_el + (n_cell - 1) * 2 * L_m + L_er) / 2

    # calculate angles outside loop
    # CALCULATE x1_el, y1_el, x2_el, y2_el
    data = ([0 + L_bp_l, Ri_el + b_el, L_el + L_bp_l, Req_el - B_el],
            [a_el, b_el, A_el, B_el])  # data = ([h, k, p, q], [a_m, b_m, A_m, B_m])

    x1el, y1el, x2el, y2el = fsolve(f, np.array(
        [a_el + L_bp_l, Ri_el + 0.85 * b_el, L_el - A_el + L_bp_l, Req_el - 0.85 * B_el]),
                                    args=data,
                                    xtol=1.49012e-12)  # [a_m, b_m-0.3*b_m, L_m-A_m, Req_m-0.7*B_m] initial guess

    # CALCULATE x1, y1, x2, y2
    data = ([0 + L_bp_l, Ri_m + b_m, L_m + L_bp_l, Req_m - B_m],
            [a_m, b_m, A_m, B_m])  # data = ([h, k, p, q], [a_m, b_m, A_m, B_m])
    x1, y1, x2, y2 = fsolve(f, np.array([a_m + L_bp_l, Ri_m + 0.85 * b_m, L_m - A_m + L_bp_l, Req_m - 0.85 * B_m]),
                            args=data, xtol=1.49012e-12)  # [a_m, b_m-0.3*b_m, L_m-A_m, Req_m-0.7*B_m] initial guess

    # CALCULATE x1_er, y1_er, x2_er, y2_er
    data = ([0 + L_bp_r, Ri_er + b_er, L_er + L_bp_r, Req_er - B_er],
            [a_er, b_er, A_er, B_er])  # data = ([h, k, p, q], [a_m, b_m, A_m, B_m])
    x1er, y1er, x2er, y2er = fsolve(f, np.array(
        [a_er + L_bp_r, Ri_er + 0.85 * b_er, L_er - A_er + L_bp_r, Req_er - 0.85 * B_er]),
                                    args=data,
                                    xtol=1.49012e-12)  # [a_m, b_m-0.3*b_m, L_m-A_m, Req_m-0.7*B_m] initial guess
    default_folder = "."

    if folder is None:
        folder = default_folder

    if name is None:
        name = 'geodata'

    with open(fr'{folder}\{name}.n', 'w') as fil:
        # SHIFT POINT TO START POINT
        start_point = [-shift, 0]
        fil.write(f"  {start_point[1]:.7E}  {start_point[0]:.7E}\n")

        lineTo(start_point, [-shift, Ri_el], step, plot=plot)
        pt = [-shift, Ri_el]
        fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")

        # ADD BEAM PIPE LENGTH
        if L_bp_l != 0:
            lineTo(pt, [L_bp_l - shift, Ri_el], step, plot=plot)
            pt = [L_bp_l - shift, Ri_el]
            print(pt)
            fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")

        for n in range(1, n_cell + 1):
            if n == 1:
                # DRAW ARC:
                pts = arcTo(L_bp_l - shift, Ri_el + b_el, a_el, b_el, step, pt, [-shift + x1el, y1el], plot=plot)
                pt = [-shift + x1el, y1el]
                for pp in pts:
                    fil.write(f"  {pp[1]:.7E}  {pp[0]:.7E}\n")
                fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")

                # DRAW LINE CONNECTING ARCS
                lineTo(pt, [-shift + x2el, y2el], step, plot=plot)
                pt = [-shift + x2el, y2el]
                fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")

                # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT
                pts = arcTo(L_el + L_bp_l - shift, Req_el - B_el, A_el, B_el, step, pt, [L_bp_l + L_el - shift, Req_el], plot=plot)
                pt = [L_bp_l + L_el - shift, Req_el]
                for pp in pts:
                    fil.write(f"  {pp[1]:.7E}  {pp[0]:.7E}\n")
                fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")

                if n_cell == 1:
                    # EQUATOR ARC TO NEXT POINT
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    pts = arcTo(L_el + L_bp_l - shift, Req_er - B_er, A_er, B_er, step, [pt[0], Req_er - B_er],
                                [L_el + L_er - x2er + L_bp_l + L_bp_r - shift, Req_er], plot=plot)
                    pt = [L_el + L_er - x2er + L_bp_l + L_bp_r - shift, y2er]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            fil.write(f"  {pp[1]:.7E}  {pp[0]:.7E}\n")
                        else:
                            pass
                    fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")

                    # STRAIGHT LINE TO NEXT POINT
                    lineTo(pt, [L_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er], step, plot=plot)
                    pt = [L_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er]
                    fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")

                    # ARC
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    pts = arcTo(L_el + L_er + L_bp_l - shift, Ri_er + b_er, a_er, b_er, step, [pt[0], Ri_er],
                                [L_bp_l + L_el + L_er - shift, y1er], plot=plot)

                    pt = [L_bp_l + L_el + L_er - shift, Ri_er]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            fil.write(f"  {pp[1]:.7E}  {pp[0]:.7E}\n")
                        else:
                            pass

                    if L_bp_r != 0:
                        fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")
                    else:
                        fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")

                    # calculate new shift
                    shift = shift - (L_el + L_er)
                    # ic(shift)
                else:
                    print("if else")
                    # EQUATOR ARC TO NEXT POINT
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    pts = arcTo(L_el + L_bp_l - shift, Req_m - B_m, A_m, B_m, step, [pt[0], Req_m - B_m],
                                [L_el + L_m - x2 + L_bp_l + L_bp_r - shift, Req_m], plot=plot)
                    pt = [L_el + L_m - x2 + L_bp_l + L_bp_r - shift, y2]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            fil.write(f"  {pp[1]:.7E}  {pp[0]:.7E}\n")
                        else:
                            pass
                    fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")

                    # STRAIGHT LINE TO NEXT POINT
                    lineTo(pt, [L_el + L_m - x1 + L_bp_l + L_bp_r - shift, y1], step, plot=plot)
                    pt = [L_el + L_m - x1 + L_bp_l + L_bp_r - shift, y1]
                    fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")

                    # ARC
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    pts = arcTo(L_el + L_m + L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, [pt[0], Ri_m],
                                [L_bp_l + L_el + L_m - shift, y1], plot=plot)
                    pt = [L_bp_l + L_el + L_m - shift, Ri_m]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            fil.write(f"  {pp[1]:.7E}  {pp[0]:.7E}\n")
                        else:
                            pass

                    fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")

                    # calculate new shift
                    shift = shift - (L_el + L_m)
                    # ic(shift)

            elif n > 1 and n != n_cell:
                # DRAW ARC:
                pts = arcTo(L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt, [-shift + x1, y1], plot=plot)
                pt = [-shift + x1, y1]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.7E}  {pp[0]:.7E}\n")
                    else:
                        pass
                fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")

                # DRAW LINE CONNECTING ARCS
                lineTo(pt, [-shift + x2, y2], step, plot=plot)
                pt = [-shift + x2, y2]
                fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")

                # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT
                pts = arcTo(L_m + L_bp_l - shift, Req_m - B_m, A_m, B_m, step, pt, [L_bp_l + L_m - shift, Req_m], plot=plot)
                pt = [L_bp_l + L_m - shift, Req_m]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.7E}  {pp[0]:.7E}\n")
                    else:
                        pass
                fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")

                # EQUATOR ARC TO NEXT POINT
                # half of bounding box is required,
                # start is the lower coordinate of the bounding box and end is the upper
                pts = arcTo(L_m + L_bp_l - shift, Req_m - B_m, A_m, B_m, step, [pt[0], Req_m - B_m],
                            [L_m + L_m - x2 + L_bp_l + L_bp_r - shift, Req_m], plot=plot)
                pt = [L_m + L_m - x2 + L_bp_l + L_bp_r - shift, y2]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.7E}  {pp[0]:.7E}\n")
                    else:
                        pass
                fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")

                # STRAIGHT LINE TO NEXT POINT
                lineTo(pt, [L_m + L_m - x1 + L_bp_l + L_bp_r - shift, y1], step, plot=plot)
                pt = [L_m + L_m - x1 + L_bp_l + L_bp_r - shift, y1]
                fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")

                # ARC
                # half of bounding box is required,
                # start is the lower coordinate of the bounding box and end is the upper
                pts = arcTo(L_m + L_m + L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, [pt[0], Ri_m],
                            [L_bp_l + L_m + L_m - shift, y1], plot=plot)
                pt = [L_bp_l + L_m + L_m - shift, Ri_m]
                ic(pt)
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.7E}  {pp[0]:.7E}\n")
                    else:
                        pass
                fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")

                # calculate new shift
                shift = shift - 2*L_m
            else:
                print("else")
                # DRAW ARC:
                pts = arcTo(L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt, [-shift + x1, y1], plot=plot)
                pt = [-shift + x1, y1]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.7E}  {pp[0]:.7E}\n")
                    else:
                        pass
                fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")

                # DRAW LINE CONNECTING ARCS
                lineTo(pt, [-shift + x2, y2], step, plot=plot)
                pt = [-shift + x2, y2]
                fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")

                # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT
                pts = arcTo(L_m + L_bp_l - shift, Req_m - B_m, A_m, B_m, step, pt, [L_bp_l + L_m - shift, Req_m], plot=plot)
                pt = [L_bp_l + L_m - shift, Req_m]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.7E}  {pp[0]:.7E}\n")
                    else:
                        pass
                fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")

                # EQUATOR ARC TO NEXT POINT
                # half of bounding box is required,
                # start is the lower coordinate of the bounding box and end is the upper
                pts = arcTo(L_m + L_bp_l - shift, Req_er - B_er, A_er, B_er, step, [pt[0], Req_er - B_er],
                            [L_m + L_er - x2er + L_bp_l + L_bp_r - shift, Req_er], plot=plot)
                pt = [L_m + L_er - x2er + L_bp_l + L_bp_r - shift, y2er]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.7E}  {pp[0]:.7E}\n")
                    else:
                        pass
                fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")

                # STRAIGHT LINE TO NEXT POINT
                lineTo(pt, [L_m + L_er - x1er + L_bp_l + L_bp_r - shift, y1er], step, plot=plot)
                pt = [L_m + L_er - x1er + L_bp_l + L_bp_r - shift, y1er]
                fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")

                # ARC
                # half of bounding box is required,
                # start is the lower coordinate of the bounding box and end is the upper
                pts = arcTo(L_m + L_er + L_bp_l - shift, Ri_er + b_er, a_er, b_er, step, [pt[0], Ri_er],
                            [L_bp_l + L_m + L_er - shift, y1er], plot=plot)
                pt = [L_bp_l + L_m + L_er - shift, Ri_er]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.7E}  {pp[0]:.7E}\n")
                    else:
                        pass

                if L_bp_r != 0:
                    fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")
                else:
                    fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")

        # BEAM PIPE
        # reset shift
        shift = (L_bp_r + L_bp_l + (n_cell - 1) * 2 * L_m + L_el + L_er) / 2

        if L_bp_r != 0:  # if there's a problem, check here.
            lineTo(pt, [L_bp_r + L_bp_l + 2 * (n_cell-1) * L_m + L_el + L_er - shift, Ri_er], step, plot=plot)
            pt = [2 * (n_cell-1) * L_m + L_el + L_er + L_bp_l + L_bp_r - shift, Ri_er]
            fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")
            print("pt after", pt)

        # END PATH
        lineTo(pt, [2 * (n_cell-1) * L_m + L_el + L_er + L_bp_l + L_bp_r - shift, 0], step, plot=plot)  # to add beam pipe to right
        pt = [2 * (n_cell-1) * L_m + L_el + L_er + L_bp_l + L_bp_r - shift, 0]
        # lineTo(pt, [2 * n_cell * L_er + L_bp_l - shift, 0], step)
        # pt = [2 * n_cell * L_er + L_bp_l - shift, 0]
        fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")

        # # CLOSE PATH
        # lineTo(pt, start_point, step, plot=plot)
        # fil.write(f"  {start_point[1]:.7E}  {start_point[0]:.7E}\n")
    if plot:
        plt.gca().set_aspect('equal', 'box')
        plt.show()

#
# def write_parallel_plate_capacitor(folder=None, name=None):
#     default_folder = "."
#
#     if folder is None:
#         folder = default_folder
#
#     if name is None:
#         name = 'geodata'
#
#     with open(fr'{folder}\{name}.n', 'w') as fil:
#         # SHIFT POINT TO START POINT
#         start_point = [-shift, 0]
#         fil.write(f"  {start_point[1]:.7E}  {start_point[0]:.7E}\n")


def write_ell_cavity_flat_top(folder=None, mid_cell=None, lend_cell=None, rend_cell=None, name=None, step=None, n_cell=None):
    if mid_cell is None and lend_cell is None and rend_cell is None:
        A_m, B_m, a_m, b_m, Ri_m, L_m, Req_m, alpha, lft =(np.array([64.453596, 54.579114, 19.1, 25.922107, 65, 83.553596, 163.975, 20]) * 1e-3)
        A_el, B_el, a_el, b_el, Ri_el, L_el, Req_el, alpha_el, lft_el = \
            (np.array([64.453596, 54.579114, 19.1, 25.922107, 65, 83.553596, 163.975, 11.187596]) * 1e-3)
        A_er, B_er, a_er, b_er, Ri_er, L_er, Req_er, alpha_er, lft_er = \
            (np.array([64.453596, 54.579114, 19.1, 25.922107, 65, 83.553596, 163.975, 11.187596]) * 1e-3)
    else:
        if lend_cell is None and rend_cell is None:
            A_m, B_m, a_m, b_m, Ri_m, L_m, Req_m, lft = np.array(mid_cell)
            A_el, B_el, a_el, b_el, Ri_el, L_el, Req_el, lft_el = np.array(mid_cell)
            A_er, B_er, a_er, b_er, Ri_er, L_er, Req_er, lft_er = np.array(mid_cell)
        elif mid_cell is None and lend_cell is None:
            A_m, B_m, a_m, b_m, Ri_m, L_m, Req_m, lft = np.array(rend_cell)
            A_el, B_el, a_el, b_el, Ri_el, L_el, Req_el, lft_el = np.array(rend_cell)
            A_er, B_er, a_er, b_er, Ri_er, L_er, Req_er, lft_er = np.array(rend_cell)
        elif mid_cell is None and rend_cell is None:
            A_m, B_m, a_m, b_m, Ri_m, L_m, Req_m, lft = np.array(lend_cell)
            A_el, B_el, a_el, b_el, Ri_el, L_el, Req_el, lft_el = np.array(lend_cell)
            A_er, B_er, a_er, b_er, Ri_er, L_er, Req_er, lft_er = np.array(lend_cell)
        else:
            print("There is something wrong with the geometry definition. Reverts to the default TESLA geometry.")
            A_m, B_m, a_m, b_m, Ri_m, L_m, Req_m, lft = \
                (np.array([64.453596, 54.579114, 19.1, 25.922107, 65, 83.553596, 163.975, 20]) * 1e-3)
            A_el, B_el, a_el, b_el, Ri_el, L_el, Req_el, lft_el = \
                (np.array([64.453596, 54.579114, 19.1, 25.922107, 65, 83.553596, 163.975, 11.187596]) * 1e-3)
            A_er, B_er, a_er, b_er, Ri_er, L_er, Req_er, lft_er = \
                (np.array([64.453596, 54.579114, 19.1, 25.922107, 65, 83.553596, 163.975, 11.187596]) * 1e-3)
    plt.rcParams["figure.figsize"] = (12, 3)

    if n_cell is None:
        n_cell = 1
    if step is None:
        step = 0.005 * min(L_m, L_el, L_er)  # step in boundary points in mm
    L_bp_l = 0.000
    L_bp_r = 0.000

    # calculate shift
    shift = (L_bp_r + L_bp_l + L_el + lft_el + (n_cell - 1) * 2 * L_m + (n_cell - 2)*lft + L_er + lft_er) / 2

    # calculate angles outside loop
    # CALCULATE x1_el, y1_el, x2_el, y2_el
    data = ([0 + L_bp_l, Ri_el + b_el, L_el + L_bp_l, Req_el - B_el],
            [a_el, b_el, A_el, B_el])  # data = ([h, k, p, q], [a_m, b_m, A_m, B_m])

    x1el, y1el, x2el, y2el = fsolve(f, np.array(
        [a_el + L_bp_l, Ri_el + 0.85 * b_el, L_el - A_el + L_bp_l, Req_el - 0.85 * B_el]),
                                    args=data, fprime=jac,
                                    xtol=1.49012e-12)  # [a_m, b_m-0.3*b_m, L_m-A_m, Req_m-0.7*B_m] initial guess

    # CALCULATE x1, y1, x2, y2
    data = ([0 + L_bp_l, Ri_m + b_m, L_m + L_bp_l, Req_m - B_m],
            [a_m, b_m, A_m, B_m])  # data = ([h, k, p, q], [a_m, b_m, A_m, B_m])
    x1, y1, x2, y2 = fsolve(f, np.array([a_m + L_bp_l, Ri_m + 0.85 * b_m, L_m - A_m + L_bp_l, Req_m - 0.85 * B_m]),
                            args=data, fprime=jac, xtol=1.49012e-12)

    # CALCULATE x1_er, y1_er, x2_er, y2_er
    data = ([0 + L_bp_r, Ri_er + b_er, L_er + L_bp_r, Req_er - B_er],
            [a_er, b_er, A_er, B_er])  # data = ([h, k, p, q], [a_m, b_m, A_m, B_m])
    x1er, y1er, x2er, y2er = fsolve(f, np.array(
        [a_er + L_bp_r, Ri_er + 0.85 * b_er, L_er - A_er + L_bp_r, Req_er - 0.85 * B_er]),
                                    args=data, fprime=jac,
                                    xtol=1.49012e-12)

    default_folder = "."

    if folder is None:
        folder = default_folder
    with open(fr'{folder}\geodata.n', 'w') as fil:
        fil.write("   2.0000000e-03   0.0000000e+00   0.0000000e+00   0.0000000e+00\n")
        fil.write("   1.25000000e-02   0.0000000e+00   0.0000000e+00   0.0000000e+00\n")  # a point inside the structure
        fil.write("  -3.1415927e+00  -2.7182818e+00   0.0000000e+00   0.0000000e+00\n")  # a point outside the structure

        # SHIFT POINT TO START POINT
        start_point = [-shift, 0]
        fil.write(f"  {start_point[1]:.7E}  {start_point[0]:.7E}\n")

        lineTo(start_point, [-shift, Ri_el], step)
        pt = [-shift, Ri_el]
        fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")

        # ADD BEAM PIPE LENGTH
        if L_bp_l != 0:
            lineTo(pt, [L_bp_l - shift, Ri_el], step)
            pt = [L_bp_l - shift, Ri_el]
            fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")

        for n in range(1, n_cell + 1):
            if n == 1:
                # DRAW ARC:
                pts = arcTo(L_bp_l - shift, Ri_el + b_el, a_el, b_el, step, pt, [-shift + x1el, y1el])
                pt = [-shift + x1el, y1el]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.7E}  {pp[0]:.7E}\n")
                    else:
                        pass
                fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")

                # DRAW LINE CONNECTING ARCS
                lineTo(pt, [-shift + x2el, y2el], step)
                pt = [-shift + x2el, y2el]
                fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")

                # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT
                pts = arcTo(L_el + L_bp_l - shift, Req_el - B_el, A_el, B_el, step, pt, [L_bp_l + L_el - shift, Req_el])
                pt = [L_bp_l + L_el - shift, Req_el]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.7E}  {pp[0]:.7E}\n")
                    else:
                        pass
                fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")

                # flat top
                lineTo(pt, [L_bp_l + L_el + lft_el - shift, Req_el], step)
                pt = [L_bp_l + L_el + lft_el - shift, Req_el]
                fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")

                if n_cell == 1:
                    # EQUATOR ARC TO NEXT POINT
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    pts = arcTo(L_el + L_bp_l + lft_el - shift, Req_er - B_er, A_er, B_er, step, [pt[0], Req_er - B_er],
                                [L_el + lft_el + L_er - x2er + L_bp_l + L_bp_r - shift, Req_er])
                    pt = [L_el + lft_el + L_er - x2er + L_bp_l + L_bp_r - shift, y2er]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            fil.write(f"  {pp[1]:.7E}  {pp[0]:.7E}\n")
                        else:
                            pass
                    fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")

                    # STRAIGHT LINE TO NEXT POINT
                    lineTo(pt, [L_el + lft_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er], step)
                    pt = [L_el + lft_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er]
                    fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")

                    # ARC
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    pts = arcTo(L_el + lft_el + L_er + L_bp_l - shift, Ri_er + b_er, a_er, b_er, step, [pt[0], Ri_er],
                                [L_bp_l + L_el + lft_el + L_er - shift, y1er])

                    pt = [L_bp_l + lft_el + L_el + L_er - shift, Ri_er]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            fil.write(f"  {pp[1]:.7E}  {pp[0]:.7E}\n")
                        else:
                            pass

                    fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")

                    # calculate new shift
                    shift = shift - (L_el + L_er + lft_el)
                else:
                    # EQUATOR ARC TO NEXT POINT
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    pts = arcTo(L_el + L_bp_l + lft_el - shift, Req_m - B_m, A_m, B_m, step, [pt[0], Req_m - B_m],
                                [L_el + lft_el + L_m - x2 + L_bp_l + L_bp_r - shift, Req_m])
                    pt = [L_el + lft_el + L_m - x2 + L_bp_l + L_bp_r - shift, y2]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            fil.write(f"  {pp[1]:.7E}  {pp[0]:.7E}\n")
                        else:
                            pass
                    fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")

                    # STRAIGHT LINE TO NEXT POINT
                    lineTo(pt, [L_el + lft_el + L_m - x1 + L_bp_l + L_bp_r - shift, y1], step)
                    pt = [L_el + lft_el + L_m - x1 + L_bp_l + L_bp_r - shift, y1]
                    fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")

                    # ARC
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    pts = arcTo(L_el + lft_el + L_m + L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, [pt[0], Ri_m],
                                [L_bp_l + L_el + lft_el + L_m - shift, y1])
                    pt = [L_bp_l + L_el + lft_el + L_m - shift, Ri_m]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            fil.write(f"  {pp[1]:.7E}  {pp[0]:.7E}\n")
                        else:
                            pass
                    fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")

                    # calculate new shift
                    shift = shift - (L_el + L_m + lft_el)
                    # ic(shift)

            elif n > 1 and n != n_cell:
                print("elif")
                # DRAW ARC:
                pts = arcTo(L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt, [-shift + x1, y1])
                pt = [-shift + x1, y1]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.7E}  {pp[0]:.7E}\n")
                    else:
                        pass
                fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")

                # DRAW LINE CONNECTING ARCS
                lineTo(pt, [-shift + x2, y2], step)
                pt = [-shift + x2, y2]
                fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")

                # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT
                pts = arcTo(L_m + L_bp_l - shift, Req_m - B_m, A_m, B_m, step, pt, [L_bp_l + L_m - shift, Req_m])
                pt = [L_bp_l + L_m - shift, Req_m]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.7E}  {pp[0]:.7E}\n")
                    else:
                        pass
                fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")

                # flat top
                lineTo(pt, [L_bp_l + L_m + lft - shift, Req_m], step)
                pt = [L_bp_l + L_el + lft - shift, Req_el]
                fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")

                # EQUATOR ARC TO NEXT POINT
                # half of bounding box is required,
                # start is the lower coordinate of the bounding box and end is the upper
                pts = arcTo(L_m + L_bp_l + lft - shift, Req_m - B_m, A_m, B_m, step, [pt[0], Req_m - B_m],
                            [L_m + L_m + lft - x2 + L_bp_l + L_bp_r - shift, Req_m])
                pt = [L_m + L_m + lft - x2 + L_bp_l + L_bp_r - shift, y2]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.7E}  {pp[0]:.7E}\n")
                    else:
                        pass
                fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")

                # STRAIGHT LINE TO NEXT POINT
                lineTo(pt, [L_m + L_m + lft - x1 + L_bp_l + L_bp_r - shift, y1], step)
                pt = [L_m + L_m + lft - x1 + L_bp_l + L_bp_r - shift, y1]
                fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")

                # ARC
                # half of bounding box is required,
                # start is the lower coordinate of the bounding box and end is the upper
                pts = arcTo(L_m + L_m + lft + L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, [pt[0], Ri_m],
                            [L_bp_l + L_m + L_m + lft - shift, y1])
                pt = [L_bp_l + L_m + L_m + lft - shift, Ri_m]
                ic(pt)
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.7E}  {pp[0]:.7E}\n")
                    else:
                        pass
                fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")

                # calculate new shift
                shift = shift - 2*L_m - (lft_el + lft)
            else:
                print("else")
                # DRAW ARC:
                pts = arcTo(L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt, [-shift + x1, y1])
                pt = [-shift + x1, y1]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.7E}  {pp[0]:.7E}\n")
                    else:
                        pass
                fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")

                # DRAW LINE CONNECTING ARCS
                lineTo(pt, [-shift + x2, y2], step)
                pt = [-shift + x2, y2]
                fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")

                # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT
                pts = arcTo(L_m + L_bp_l - shift, Req_m - B_m, A_m, B_m, step, pt, [L_bp_l + L_m - shift, Req_m])
                pt = [L_bp_l + L_m - shift, Req_m]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.7E}  {pp[0]:.7E}\n")
                    else:
                        pass
                fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")

                # flat top
                lineTo(pt, [L_bp_l + L_m + lft_er - shift, Req_m], step)
                pt = [L_bp_l + L_m + lft_er - shift, Req_m, Req_el]
                fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")

                # EQUATOR ARC TO NEXT POINT
                # half of bounding box is required,
                # start is the lower coordinate of the bounding box and end is the upper
                pts = arcTo(L_m + lft_er + L_bp_l - shift, Req_er - B_er, A_er, B_er, step, [pt[0], Req_er - B_er],
                            [L_m + L_er + lft_er - x2er + L_bp_l + L_bp_r - shift, Req_er])
                pt = [L_m + L_er + lft_er - x2er + L_bp_l + L_bp_r - shift, y2er]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.7E}  {pp[0]:.7E}\n")
                    else:
                        pass
                fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")

                # STRAIGHT LINE TO NEXT POINT
                lineTo(pt, [L_m + L_er + lft_er - x1er + L_bp_l + L_bp_r - shift, y1er], step)
                pt = [L_m + L_er + lft_er - x1er + L_bp_l + L_bp_r - shift, y1er]
                fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")

                # ARC
                # half of bounding box is required,
                # start is the lower coordinate of the bounding box and end is the upper
                pts = arcTo(L_m + L_er + lft_er + L_bp_l - shift, Ri_er + b_er, a_er, b_er, step, [pt[0], Ri_er],
                            [L_bp_l + L_m + L_er + lft_er - shift, y1er])
                pt = [L_bp_l + L_m + L_er + lft_er - shift, Ri_er]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.7E}  {pp[0]:.7E}\n")
                    else:
                        pass
                fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")

        # BEAM PIPE
        # reset shift
        shift = (L_bp_r + L_bp_l + L_el + lft_el + (n_cell - 1) * 2 * L_m + (n_cell - 2)*lft + L_er + lft_er) / 2
        lineTo(pt, [L_bp_r + L_bp_l + 2 * (n_cell-1) * L_m + (n_cell-2)*lft + lft_el + lft_er + L_el + L_er - shift, Ri_er], step)

        if L_bp_r != 0:
            pt = [2 * (n_cell-1) * L_m + L_el + L_er + L_bp_l + L_bp_r + (n_cell-2)*lft + lft_el + lft_er - shift, Ri_er]
            fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}\n")
            print("pt after", pt)

        # END PATH
        lineTo(pt, [2 * (n_cell-1) * L_m + L_el + L_er + (n_cell-2)*lft + lft_el + lft_er + L_bp_l + L_bp_r - shift, 0], step)  # to add beam pipe to right
        pt = [2 * (n_cell-1) * L_m + L_el + L_er + (n_cell-2)*lft + lft_el + lft_er + L_bp_l + L_bp_r - shift, 0]
        # lineTo(pt, [2 * n_cell * L_er + L_bp_l - shift, 0], step)
        # pt = [2 * n_cell * L_er + L_bp_l - shift, 0]
        fil.write(f"  {pt[1]:.7E}  {pt[0]:.7E}   0.0000000e+00   0.0000000e+00\n")

        # CLOSE PATH
        lineTo(pt, start_point, step)
        fil.write(f"  {start_point[1]:.7E}  {start_point[0]:.7E}   0.0000000e+00   0.0000000e+00\n")

    plt.show()


def f(z, *data):
    """
    Calculates the coordinates of the tangent line that connects two ellipses

    .. _ellipse tangent:

    .. figure:: ./images/ellipse_tangent_.png
       :alt: ellipse tangent
       :align: center
       :width: 400px

    Parameters
    ----------
    z: list, array like
        Contains list of tangent points coordinate's variables ``[x1, y1, x2, y2]``.
    data: list, array like
        Contains midpoint coordinates of the two ellipses and the dimensions of the ellipses
        data = ``[coords, dim]``; ``coords`` = ``[h, k, p, q]``, ``dim`` = ``[a, b, A, B]``


    Returns
    -------
    list of four non-linear functions

    Note
    -----
    The four returned non-linear functions are

    .. math::

       f_1 = \\frac{A^2b^2(x_1 - h)(y_2-q)}{a^2B^2(x_2-p)(y_1-k)} - 1

       f_2 = \\frac{(x_1 - h)^2}{a^2} + \\frac{(y_1-k)^2}{b^2} - 1

       f_3 = \\frac{(x_2 - p)^2}{A^2} + \\frac{(y_2-q)^2}{B^2} - 1

       f_4 = \\frac{-b^2(x_1-x_2)(x_1-h)}{a^2(y_1-y_2)(y_1-k)} - 1
    """

    coord, dim = data
    h, k, p, q = coord
    a, b, A, B = dim
    x1, y1, x2, y2 = z

    f1 = A ** 2 * b ** 2 * (x1 - h) * (y2 - q) / (a ** 2 * B ** 2 * (x2 - p) * (y1 - k)) - 1
    f2 = (x1 - h) ** 2 / a ** 2 + (y1 - k) ** 2 / b ** 2 - 1
    f3 = (x2 - p) ** 2 / A ** 2 + (y2 - q) ** 2 / B ** 2 - 1
    f4 = -b ** 2 * (x1 - x2) * (x1 - h) / (a ** 2 * (y1 - y2) * (y1 - k)) - 1

    return f1, f2, f3, f4


def jac(z, *data):
    """
    Computes the Jacobian of the non-linear system of ellipse tangent equations

    Parameters
    ----------
    z: list, array like
        Contains list of tangent points coordinate's variables ``[x1, y1, x2, y2]``.
    data: list, array like
        Contains midpoint coordinates of the two ellipses and the dimensions of the ellipses
        data = ``[coords, dim]``; ``coords`` = ``[h, k, p, q]``, ``dim`` = ``[a, b, A, B]``

    Returns
    -------
    J: array like
        Array of the Jacobian

    """
    coord, dim = data
    h, k, p, q = coord
    a, b, A, B = dim
    x1, y1, x2, y2 = z

    # f1 = A ** 2 * b ** 2 * (x1 - h) * (y2 - q) / (a ** 2 * B ** 2 * (x2 - p) * (y1 - k)) - 1
    # f2 = (x1 - h) ** 2 / a ** 2 + (y1 - k) ** 2 / b ** 2 - 1
    # f3 = (x2 - p) ** 2 / A ** 2 + (y2 - q) ** 2 / B ** 2 - 1
    # f4 = -b ** 2 * (x1 - x2) * (x1 - h) / (a ** 2 * (y1 - y2) * (y1 - k)) - 1

    df1_dx1 = A ** 2 * b ** 2 * (y2 - q) / (a ** 2 * B ** 2 * (x2 - p) * (y1 - k))
    df1_dy1 = - A ** 2 * b ** 2 * (x1 - h) * (y2 - q) / (a ** 2 * B ** 2 * (x2 - p) * (y1 - k)**2)
    df1_dx2 = - A ** 2 * b ** 2 * (x1 - h) * (y2 - q) / (a ** 2 * B ** 2 * (x2 - p)**2 * (y1 - k))
    df1_dy2 = A ** 2 * b ** 2 * (x1 - h) / (a ** 2 * B ** 2 * (x2 - p) * (y1 - k))

    df2_dx1 = 2 * (x1 - h) / a ** 2
    df2_dy1 = 2 * (y1 - k) / b ** 2
    df2_dx2 = 0
    df2_dy2 = 0

    df3_dx1 = 0
    df3_dy1 = 0
    df3_dx2 = 2 * (x2 - p) / A ** 2
    df3_dy2 = 2 * (y2 - q) / B ** 2

    df4_dx1 = -b ** 2 * ((x1 - x2) + (x1 - h)) / (a ** 2 * (y1 - y2) * (y1 - k))
    df4_dy1 = -b ** 2 * (x1 - x2) * (x1 - h) * ((y1 - y2) + (y1 - k)) / (a ** 2 * ((y1 - y2) * (y1 - k))**2)
    df4_dx2 = b ** 2 * (x1 - h) / (a ** 2 * (y1 - y2) * (y1 - k))
    df4_dy2 = -b ** 2 * (x1 - x2) * (x1 - h) / (a ** 2 * (y1 - y2)**2 * (y1 - k))

    J = [[df1_dx1, df1_dy1, df1_dx2, df1_dy2],
         [df2_dx1, df2_dy1, df2_dx2, df2_dy2],
         [df3_dx1, df3_dy1, df3_dx2, df3_dy2],
         [df4_dx1, df4_dy1, df4_dx2, df4_dy2]]

    return J


def linspace(start, stop, step=1.):
    """
    Like np.linspace but uses step instead of num
    This is inclusive to stop, so if start=1, stop=3, step=0.5
    Output is: array([1., 1.5, 2., 2.5, 3.])
    """
    if start < stop:
        ll = np.linspace(start, stop, int(np.ceil((stop - start) / abs(step) + 1)))
        if stop not in ll:
            ll = np.append(ll, stop)

        return ll
    else:
        ll = np.linspace(stop, start, int(np.ceil((start - stop) / abs(step) + 1)))
        if start not in ll:
            ll = np.append(ll, start)
        return ll


def lineTo(prevPt, nextPt, step, plot=False):
    if prevPt[0] == nextPt[0]:
        # vertical line
        # check id nextPt is greater
        if prevPt[1] < nextPt[1]:
            py = linspace(prevPt[1], nextPt[1], step)
        else:
            py = linspace(nextPt[1], prevPt[1], step)
            py = py[::-1]
        px = np.ones(len(py)) * prevPt[0]

    elif prevPt[1] == nextPt[1]:
        # horizontal line
        if prevPt[0] < nextPt[1]:
            px = linspace(prevPt[0], nextPt[0], step)
        else:
            px = linspace(nextPt[0], prevPt[0], step)

        py = np.ones(len(px)) * prevPt[1]
    else:
        # calculate angle to get appropriate step size for x and y
        ang = np.arctan((nextPt[1] - prevPt[1]) / (nextPt[0] - prevPt[0]))
        if prevPt[0] < nextPt[0] and prevPt[1] < nextPt[1]:
            px = linspace(prevPt[0], nextPt[0], step * np.cos(ang))
            py = linspace(prevPt[1], nextPt[1], step * np.sin(ang))
        elif prevPt[0] > nextPt[0] and prevPt[1] < nextPt[1]:
            px = linspace(nextPt[0], prevPt[0], step * np.cos(ang))
            px = px[::-1]
            py = linspace(prevPt[1], nextPt[1], step * np.sin(ang))
        elif prevPt[0] < nextPt[0] and prevPt[1] > nextPt[1]:
            px = linspace(prevPt[0], nextPt[0], step * np.cos(ang))
            py = linspace(nextPt[1], prevPt[1], step * np.sin(ang))
            py = py[::-1]
        else:
            px = linspace(nextPt[0], prevPt[0], step * np.cos(ang))
            px = px[::-1]
            py = linspace(nextPt[1], prevPt[1], step * np.sin(ang))
            py = py[::-1]
    if plot:
        plt.plot(px, py, marker='x') #

    return np.array([px, py]).T


def arcTo(x_center, y_center, a, b, step, start, end, plot=False):
    u = x_center  # <- x-position of the center
    v = y_center  # <- y-position of the center
    a = a  # <- radius on the x-axis
    b = b  # <- radius on the y-axis
    C = np.pi*(a+b)  # <- approximate perimeter of ellipse

    t = np.arange(0, 2 * np.pi, np.pi / int(np.ceil(C/step)))

    x = u + a * np.cos(t)
    y = v + b * np.sin(t)
    pts = np.column_stack((x, y))
    inidx = np.all(np.logical_and(np.array(start) < pts, pts < np.array(end)), axis=1)
    inbox = pts[inidx]
    inbox = inbox[inbox[:, 0].argsort()]

    if plot:
        plt.plot(inbox[:, 0], inbox[:, 1], marker='x')#

    return inbox





















