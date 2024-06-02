import copy

from ngsolve import *
from ngsolve.webgui import Draw
from netgen.occ import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import scipy

q0 = 1.60217663e-19
m0 = 9.1093837e-31
mu0 = 4 * np.pi * 1e-7
eps0 = 8.85418782e-12
c0 = 299792458


class Integrators:
    def __init__(self, mesh, w, bounding_rect):
        self.mesh = mesh
        self.w = w

        self.zmin, self.zmax, self.rmin, self.rmax = bounding_rect

    def forward_euler(self, particles, tn, h, em, scale, sey):
        ku1 = h * self.lorentz_force(particles, tn, em, scale)
        particles.u = particles.u + ku1
        particles.x = particles.x + h * particles.u
        self.plot_path(particles)

        # check for lost particles
        lpi, rpi = self.hit_bound(particles, tn, h, em, scale, sey)

        if len(rpi) != 0:
            particles.update_hit_count(rpi)

        if len(lpi) != 0:
            particles.remove(lpi)

    def implicit_euler(self, particles, tn, h, em, scale):
        pass

    def rk2(self, particles, tn, h, em, scale):

        # k1
        ku1 = h * self.lorentz_force(particles, tn, em, scale)
        kx1 = h * particles.u

        particles_dummy = copy.deepcopy(particles)
        particles_dummy.save_old()
        particles_dummy.u += ku1 / 2
        particles_dummy.x += kx1 / 2

        # check for lost particles
        lpi, rpi = self.hit_bound(particles_dummy, tn + h / 2, h / 2, em, scale)

        for rp in rpi:
            ku1[rp] = particles_dummy.u[rp] - particles.u[rp]
            kx1[rp] = particles_dummy.x[rp] - particles.x[rp]

        if len(lpi) != 0:
            [ku1], [kx1] = self.rk_update_k([ku1], [kx1], lpi)
            particles_dummy.remove(lpi)
            particles.remove(lpi)
            em.remove(lpi)

        # check if all particles are lost
        if particles.len == 0:
            return False
        # k2=================================================
        ku2 = h * self.lorentz_force(particles_dummy, tn + 2 * h / 3, em,
                                     scale)  # <- particles dummy = particles.u + kn
        kx2 = h * (particles.u + 2 * ku1 / 3)

        particles_dummy = copy.deepcopy(particles)
        particles_dummy.save_old()
        particles_dummy.u += 2 * ku2 / 3
        particles_dummy.x += 2 * kx2 / 3

        # check for lost particles
        lpi, rpi = self.hit_bound(particles_dummy, tn + 2 * h / 3, 2 * h / 3, em, scale)

        for rp in rpi:
            ku2[rp] = particles_dummy.u[rp] - particles.u[rp]
            kx2[rp] = particles_dummy.x[rp] - particles.x[rp]

        if len(lpi) != 0:
            particles_dummy.remove(lpi)
            [ku1, ku2], [kx1, kx2] = self.rk_update_k([ku1, ku2], [kx1, kx2], lpi)
            particles.remove(lpi)
            em.remove(lpi)

        particles.u = particles.u + ku2
        particles.x = particles.x + kx1

        # check for lost particles
        lpi, rpi = self.hit_bound(particles, tn, h, em, scale)
        if len(lpi) != 0:
            particles.remove(lpi)
            em.remove(lpi)

        # check if all particles are lost
        if particles.len == 0:
            return False
        self.plot_path(particles)

    def rk2_23(self, particles, tn, h, em, scale):
        # k1
        ku1 = h * self.lorentz_force(particles, tn, em, scale)
        kx1 = h * particles.u

        particles_dummy = copy.deepcopy(particles)
        particles_dummy.save_old()
        particles_dummy.u += ku1 / 2
        particles_dummy.x += kx1 / 2

        # check for lost particles
        lpi, rpi = self.hit_bound(particles_dummy, tn + h / 2, h / 2, em, scale)

        for rp in rpi:
            ku1[rp] = particles_dummy.u[rp] - particles.u[rp]
            kx1[rp] = particles_dummy.x[rp] - particles.x[rp]
            particles.update_hit_count(rpi)

        if len(lpi) != 0:
            [ku1], [kx1] = self.rk_update_k([ku1], [kx1], lpi)
            particles_dummy.remove(lpi)
            particles.remove(lpi)
            em.remove(lpi)

        # check if all particles are lost
        if particles.len == 0:
            return False
        # k2=================================================
        ku2 = h * self.lorentz_force(particles_dummy, tn + h / 2, em, scale)  # <- particles dummy = particles.u + kn
        kx2 = h * (particles.u + ku1 / 2)

        particles_dummy = copy.deepcopy(particles)
        particles_dummy.save_old()
        particles_dummy.u += ku2 / 2
        particles_dummy.x += kx2 / 2

        # check for lost particles
        lpi, rpi = self.hit_bound(particles_dummy, tn + h / 2, h / 2, em, scale)

        for rp in rpi:
            ku2[rp] = particles_dummy.u[rp] - particles.u[rp]
            kx2[rp] = particles_dummy.x[rp] - particles.x[rp]
            particles.update_hit_count(rpi)

        if len(lpi) != 0:
            particles_dummy.remove(lpi)
            [ku1, ku2], [kx1, kx2] = self.rk_update_k([ku1, ku2], [kx1, kx2], lpi)
            particles.remove(lpi)
            em.remove(lpi)

        particles.u = particles.u + (1 / 4 * ku1 + 3 / 4 * ku2)
        particles.x = particles.x + (1 / 4 * kx1 + 3 / 4 * kx2)

        # check for lost particles
        lpi, rpi = self.hit_bound(particles, tn, h, em, scale)
        if len(lpi) != 0:
            particles.remove(lpi)
            em.remove(lpi)

        # check if all particles are lost
        if particles.len == 0:
            return False
        self.plot_path(particles)

    # def rk4(self, particles, tn, h, em, scale, sey):
    #     #         start = time.time()
    #     # k1
    #     ku1 = h * self.lorentz_force(particles, tn, em, scale)
    #     kx1 = h * particles.u
    #     print('Strat')
    #     #         ss = time.time()
    #     particles_dummy = copy.deepcopy(particles)
    #
    #     particles_dummy.save_old()
    #     particles_dummy.u += ku1 / 2
    #     particles_dummy.x += kx1 / 2
    #
    #     try:
    #         print('In here 1')
    #         ku2 = h * self.lorentz_force(particles_dummy, tn + h / 2, em,
    #                                      scale)  # <- particles dummy = particles.u + kn
    #         kx2 = h * (particles.u + ku1 / 2)
    #     except:
    #         # take full step - euler
    #         particles.u = particles.u + ku1
    #         particles.x = particles.x + kx1
    #
    #         lpi, rpi = self.hit_bound(particles, tn, h, em, scale, sey)
    #
    #         if len(rpi) != 0:
    #             particles.update_hit_count(list(set(rpi)))
    #
    #         if len(lpi) != 0:
    #             particles.remove(lpi)
    #
    #         self.plot_path(particles, tn)
    #         print('Return 2')
    #         return
    #
    #     particles_dummy = copy.deepcopy(particles)
    #
    #     particles_dummy.save_old()
    #     particles_dummy.u += ku2 / 2
    #     particles_dummy.x += kx2 / 2
    #
    #     try:
    #         print('In here 3')
    #         ku3 = h * self.lorentz_force(particles_dummy, tn + h / 2, em,
    #                                      scale)  # <- particles dummy = particles.u + kn
    #         kx3 = h * (particles.u + ku2 / 2)
    #     except:
    #         # take full step - euler
    #         particles.u = particles.u + ku1
    #         particles.x = particles.x + kx1
    #
    #         lpi, rpi = self.hit_bound(particles, tn, h, em, scale, sey)
    #
    #         if len(rpi) != 0:
    #             particles.update_hit_count(list(set(rpi)))
    #
    #         if len(lpi) != 0:
    #             particles.remove(lpi)
    #
    #         self.plot_path(particles, tn)
    #         print('Return 2')
    #
    #         return
    #
    #     particles_dummy = copy.deepcopy(particles)
    #
    #     particles_dummy.save_old()
    #     particles_dummy.u += ku3
    #     particles_dummy.x += kx3
    #
    #     try:
    #         print('In here 4')
    #         ku4 = h * self.lorentz_force(particles_dummy, tn + h, em, scale)  # <- particles dummy = particles.u + kn
    #         kx4 = h * particles_dummy.u
    #     except:
    #         # take full step - euler
    #         particles.u = particles.u + ku1
    #         particles.x = particles.x + kx1
    #
    #         lpi, rpi = self.hit_bound(particles, tn, h, em, scale, sey)
    #
    #         if len(rpi) != 0:
    #             particles.update_hit_count(list(set(rpi)))
    #
    #         if len(lpi) != 0:
    #             particles.remove(lpi)
    #
    #         self.plot_path(particles, tn)
    #         print('Return 3')
    #
    #         return
    #
    #     particles.u = particles.u + 1 / 6 * (ku1 + 2 * ku2 + 2 * ku3 + ku4)
    #     particles.x = particles.x + 1 / 6 * (kx1 + 2 * kx2 + 2 * kx3 + kx4)
    #
    #     # check for lost particles
    #     lpi, rpi = self.hit_bound(particles, tn, h, em, scale, sey)
    #
    #     if len(rpi) != 0:
    #         particles.update_hit_count(list(set(rpi)))
    #
    #     if len(lpi) != 0:
    #         particles.remove(lpi)
    #
    #     # check if all particles are lost
    #     if particles.len == 0:
    #         print('Return 54')
    #         return False
    #     #         print("rk4 exec time: ", time.time() - start)
    #     #         print('='*80)
    #
    #     self.plot_path(particles, tn)
    #     print('Finish')

    def rk4(self, particles, tn, h, em, scale, sey):
        lpi_all = []
        rpi_all = []
        lpi_rpi_all = []

        mask = np.ones(len(particles.x), dtype=bool)
        # k1
        ku1 = h * self.lorentz_force(particles, mask, tn, em, scale)
        kx1 = h * particles.u
        particles_dummy = copy.deepcopy(particles)
        particles_dummy.save_old()
        particles_dummy.u += ku1 / 2
        particles_dummy.x += kx1 / 2

        try:
            ku2, kx2 = np.zeros_like(particles.x), np.zeros_like(particles.x)

            ku2[mask] += h * self.lorentz_force(particles_dummy, mask, tn + h / 2, em, scale)  # <- particles dummy = particles.u + kn
            kx2[mask] += h * (particles_dummy.u[mask])
        except Exception as e:
            # print('EXCEPTION1:: ')
            lpi, rpi = self.hit_bound(particles, particles_dummy, mask, tn, h, em, scale, sey)
            lpi_all.extend(lpi)
            rpi_all.extend(rpi)
            lpi_rpi_all = lpi_all + rpi_all

            if len(lpi_rpi_all) != 0:
                mask[np.sort(lpi_rpi_all)] = False

            particles_dummy = copy.deepcopy(particles)
            particles_dummy.save_old()

            ku2, kx2 = np.zeros_like(particles.x), np.zeros_like(particles.x)
            ku2[mask] += h * self.lorentz_force(particles_dummy, mask, tn + h / 2, em,
                                                     scale)  # <- particles dummy = particles.u + kn
            kx2[mask] += h * (particles_dummy.u[mask] + ku1[mask] / 2)

        particles_dummy = copy.deepcopy(particles)
        particles_dummy.save_old()
        particles_dummy.u[mask] += ku2[mask] / 2
        particles_dummy.x[mask] += kx2[mask] / 2

        try:
            ku3, kx3 = np.zeros_like(particles.x), np.zeros_like(particles.x)
            ku3[mask] += h * self.lorentz_force(particles_dummy, mask, tn + h / 2, em,
                                                     scale)  # <- particles dummy = particles.u + kn
            kx3[mask] += h * (particles_dummy.u[mask])
        except:
            # print('EXCEPTION2:: ')
            lpi, rpi = self.hit_bound(particles, particles_dummy, mask, tn, h, em, scale, sey)
            lpi_all.extend(lpi)
            rpi_all.extend(rpi)

            lpi_rpi_all = lpi_all + rpi_all
            if len(lpi_rpi_all) != 0:
                mask[np.sort(lpi_rpi_all)] = False

            particles_dummy = copy.deepcopy(particles)
            particles_dummy.save_old()

            ku3, kx3 = np.zeros_like(particles.x), np.zeros_like(particles.x)
            ku3[mask] += h * self.lorentz_force(particles_dummy, mask, tn + h / 2, em,
                                                     scale)  # <- particles dummy = particles.u + kn
            kx3[mask] += h * (particles_dummy.u[mask] + ku1[mask] / 2)

        particles_dummy = copy.deepcopy(particles)
        particles_dummy.save_old()
        particles_dummy.u[mask] += ku3[mask]
        particles_dummy.x[mask] += kx3[mask]

        try:
            ku4, kx4 = np.zeros_like(particles.x), np.zeros_like(particles.x)
            ku4[mask] += h * self.lorentz_force(particles_dummy, mask, tn + h, em,
                                                     scale)  # <- particles dummy = particles.u + kn
            kx4[mask] += h * (particles_dummy.u[mask] + ku3[mask])
        except:
            # print('EXCEPTION3:: ', mask, len(particles.x), lpi_rpi_all)
            lpi, rpi = self.hit_bound(particles, particles_dummy, mask, tn, h, em, scale, sey)
            lpi_all.extend(lpi)
            rpi_all.extend(rpi)

            lpi_rpi_all = lpi_all + rpi_all
            # print(mask, lpi, rpi, lpi_all, rpi_all, lpi_rpi_all)
            if len(lpi_rpi_all) != 0:
                mask[np.sort(lpi_rpi_all)] = False

            particles_dummy = copy.deepcopy(particles)
            particles_dummy.save_old()

            ku4, kx4 = np.zeros_like(particles.x), np.zeros_like(particles.x)
            ku4[mask] += h * self.lorentz_force(particles_dummy, mask, tn + h, em,
                                                scale)  # <- particles dummy = particles.u + kn
            kx4[mask] += h * (particles_dummy.u[mask] + ku3[mask])

        particles_dummy = copy.deepcopy(particles)
        particles_dummy.save_old()
        particles_dummy.u[mask] += 1 / 6 * (ku1[mask] + 2 * ku2[mask] + 2 * ku3[mask] + ku4[mask])
        particles_dummy.x[mask] += 1 / 6 * (kx1[mask] + 2 * kx2[mask] + 2 * kx3[mask] + kx4[mask])

        # print("dummy particle", len(particles_dummy.x), '\n', particles_dummy.x)
        # check for lost particles
        lpi, rpi = self.hit_bound(particles, particles_dummy, mask, tn, h, em, scale, sey)
        lpi_all.extend(lpi)
        rpi_all.extend(rpi)

        lpi_rpi_all = lpi_all + rpi_all
        if len(lpi_rpi_all) != 0:
            mask[np.sort(lpi_rpi_all)] = False

        # modify to only update particles not reflected or lost
        particles.u[mask] += 1 / 6 * (ku1[mask] + 2 * ku2[mask] + 2 * ku3[mask] + ku4[mask])
        particles.x[mask] += 1 / 6 * (kx1[mask] + 2 * kx2[mask] + 2 * kx3[mask] + kx4[mask])

        removed_inds = np.array([])
        if len(rpi_all) != 0:
            removed_inds = particles.update_hit_count(list(set(rpi_all)))

        if len(lpi_all) != 0:
            particles.remove(self.update_lpi(lpi_all, removed_inds))

        # check if all particles are lost
        if particles.len == 0:
            return False
        self.plot_path(particles, tn)

        # print("Done rk4", len(particles.x), '\n', particles.x)
        # particles.trace(self.domain.ax)
        # print('=='*50)

    def rkf45(self):
        pass

    def rk_update_k(self, ku_list, kx_list, lpi):
        ku_list_new, kx_list_new = [], []
        for ku in ku_list:
            ku = np.delete(ku, lpi, axis=0)
            ku_list_new.append(ku)
        for kx in kx_list:
            kx = np.delete(kx, lpi, axis=0)
            kx_list_new.append(kx)

        return ku_list_new, kx_list_new

    def adams_bashforth(self):
        pass

    def leapfrog(self):
        pass

    def lorentz_force(self, particles, mask, tn, em, scale):
        x, u, phi = particles.x[mask], particles.u[mask], particles.phi[mask]
        # get e and b field from eigenmode analysis at particle current position
        #         print("\t\t Before lorentz")
        #         print(pos, scale)
        e = scale * em.e(self.mesh(x[:, 0], x[:, 1])) * np.exp(1j * (self.w * tn + phi))
        b = mu0 * scale * em.h(self.mesh(x[:, 0], x[:, 1])) * np.exp(1j * (self.w * tn + phi))

        k = q0 / m0 * np.sqrt(1 - (self.norm(u) / c0) ** 2) * (
                e.real + self.cross(u, b.real) - (1 / (c0 ** 2)) * (self.dot(u, e.real) * u))  # <- relativistic

        return k

    def plot_path(self, particles, tn=None):
        if tn is None:
            tn = 1 / self.w
        #     print('before', particles.paths.shape, particles.paths_count)
        #     print(particles.x.shape, particles.phi.shape, particles.paths.shape)
        particles.paths = np.vstack((particles.paths, np.hstack((particles.x, self.w * tn + particles.phi))))
        particles.paths_count += 1

    def hit_bound(self, particles, particles_dummy, mask, t, dt, em, scale, sey):
        xsurf = particles_dummy.bounds
        #     # check if particle close to boundary
        res, indx = particles_dummy.distance(50)
        ind_ = np.where(res <= c0 * dt)
        res, indx = res[ind_[0], ind_[1]], np.array(indx)[ind_[0], :]
        #     print('\t\t\t\t distance time:', time.time()-ss)

        lost_particles_indx = []
        reflected_particles_indx = []
        for ind, r, idx in zip(ind_[0], res, indx):
            #             if r < 5e-2 and mask[ind]:  # point at boundary, calculate new field value
            if r < c0 * dt and mask[ind]:  # point at boundary, calculate new field value
                # check if point is inside or outside of region
                # get surface points neighbours
                surf_pts_neigs = self.get_neighbours(xsurf, idx)

                # check for intersection
                # get intersection with old point. loop through points again.
                # the surface edge a line between an outside point and the origin intersects
                # might be different from that with which the line between old and new point intersects

                line11 = (particles_dummy.x[ind],
                          particles_dummy.x_old[ind])  # <- straight line btw current and previous points
                line22 = surf_pts_neigs[1:], surf_pts_neigs[:-1]

                bool_intc_p, x_intc_p, intc_indx = self.segment_intersection(line11, line22)
                # plt.plot(np.array(line11).T[0], np.array(line11).T[1], c='b', marker='o', zorder=2000)
                # plt.plot(np.array(line22).T[0], np.array(line22).T[1], c='r', marker='o')

                if bool_intc_p:
                    dt_frac = np.linalg.norm(x_intc_p - particles_dummy.x_old[ind]) / np.linalg.norm(
                        particles_dummy.x[ind] - particles_dummy.x_old[ind])
                    t_frac = t - dt * (1 - dt_frac)

                    # Advance particle to surface
                    #  calculate field values at this time which is a (fraction of dt) + t
                    e = scale * np.array([em.e(self.mesh(*x_intc_p))]) * np.exp(
                        1j * (self.w * t_frac + particles_dummy.phi[ind]))
                    b = mu0 * scale * np.array([em.h(self.mesh(*x_intc_p))]) * np.exp(
                        1j * (self.w * t_frac + particles_dummy.phi[ind]))

                    # check if the e-field surface normal is close to zero indicating a possible change in field
                    line22 = np.array(line22)[:, intc_indx]
                    line22 = line22[line22[:, 0].argsort()]
                    line22_normal = -np.array([-(line22[1][1] - line22[0][1]), line22[1][0] - line22[0][0]])
                    line22_normal = line22_normal / np.linalg.norm(line22_normal)
                    # plt.plot(np.array(line11).T[0], np.array(line11).T[1], c='k', marker='o', zorder=2000)
                    # plt.plot(np.array(line22).T[0], np.array(line22).T[1], c='g', marker='o')

                    e_dot_surf_norm = np.dot(e.real, line22_normal)
                    if e_dot_surf_norm >= 0:
                        particles_dummy.u_temp[ind] = (particles_dummy.u_old[ind] +
                                                       q0 / m0 * np.sqrt(1 - (self.norm([particles_dummy.u_old[ind]]) / c0) ** 2) *
                                                       (e.real + self.cross([particles_dummy.u_old[ind]], b.real) -
                                                        (1 / c0 ** 2) * (self.dot([particles_dummy.u_old[ind]], e.real) *
                                                                         particles_dummy.u_old[ind])) * dt * dt_frac)

                        # check if conditions support secondary electron yield
                        # calculate electron energy
                        umag = np.linalg.norm(particles_dummy.u_temp[ind])
                        gamma = 1 / (np.sqrt(1 - (umag / c0) ** 2))
                        # pm = gamma * m0 * umag
                        Eq = (gamma - 1) * m0 * c0 ** 2 * 6.241509e18  # 6.241509e18 Joules to eV factor

                        # update main particles array
                        particles.E[ind].append(Eq)

                        # calculate number of secondary electrons
                        if sey.Emin < Eq < sey.Emax:
                            particles.n_secondaries[ind].append(float(sey.sey(Eq)))
                        else:
                            particles.n_secondaries[ind].append(0)

                        # calculate new position using 1-dt_frac, u_temp at intersection and x_temp
                        # u_emission = line22_normal * np.sqrt(2 * particles.init_v * q0 / m0)
                        u_emission = line22_normal * particles.init_v

                        # use impact energy to calculate velocity of secondary particles
                        # to be implemented
                        particles.u[ind] = (u_emission + q0 / m0 * np.sqrt(1 - (self.norm([u_emission]) / c0) ** 2) *
                                            (e.real + self.cross([u_emission], b.real) - (1 / c0 ** 2) *
                                             (self.dot([u_emission], e.real) * u_emission)) * dt * (1 - dt_frac))

                        particles.x[ind] = x_intc_p + particles.u[ind] * dt * (1 - dt_frac)
                        reflected_particles_indx.append(ind)
                    else:
                        lost_particles_indx.append(ind)

        # finally check if particle is at the other boundaries not the wall surface
        for indx_ob, ptx in enumerate(particles_dummy.x):
            if ptx[1] <= self.rmin:  # <- bottom edge (rotation axis) check
                lost_particles_indx.append(indx_ob)
            if ptx[0] <= self.zmin or ptx[0] >= self.zmax:  # <- left and right boundaries
                lost_particles_indx.append(indx_ob)

        return lost_particles_indx, reflected_particles_indx

    @staticmethod
    def cross(a, b):
        c1 = np.array(a)[:, 1] * np.array(b)[:, 0]
        c2 = -np.array(a)[:, 0] * np.array(b)[:, 0]
        return np.array([c1, c2]).T

    @staticmethod
    def dot(a, b):
        return np.atleast_2d(np.sum(a * b, axis=1)).T

    @staticmethod
    def norm(a):
        return np.atleast_2d(np.linalg.norm(a, axis=1)).T

    @staticmethod
    def segment_intersection(line1, line2):
        x1, y1 = line1[0]
        x2, y2 = line1[1]
        x3, y3 = line2[0][:, 0], line2[0][:, 1]
        x4, y4 = line2[1][:, 0], line2[1][:, 1]

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        #     if denom == 0:
        #        return False, (0, 0)

        tt = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        uu = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / denom

        # get index of where condition is true. condition should be true at only one point but for the case that
        #  a line has one point on a surface node intersects two edges
        condition = np.where((tt >= 0) * (tt <= 1) * (uu >= 0) * (uu <= 1))[0]
        # print('condition', condition)

        if len(condition) > 0:  # review for more complex geometry. a line that has one point on a surface node intersects two points.
            px, py = x1 + tt[condition[0]] * (x2 - x1), y1 + tt[condition[0]] * (y2 - y1)
            return True, np.array([px, py]), condition[0]
        else:
            return False, np.array([0, 0]), 0

    @staticmethod
    def get_neighbours(surf_pts, idx):
        surf_pts_neigs = np.array(surf_pts[idx])
        return surf_pts_neigs[surf_pts_neigs[:, 0].argsort()]

    def collision(self, active_interval):
        pass

    def indices_corrector(self, ind1, ind2):
        array_comp = np.array(ind2)
        for ii in ind1:
            array_comp += (array_comp >= ii) * 1
        return array_comp

    def compose_indices(self, indices_array):
        composed_indices = np.array([])
        for i in range(len(indices_array) - 1):
            if i == 0:
                xx = np.array(indices_array[i])
            else:
                xx = composed_indices

            array_comp = np.array(indices_array[i + 1])
            for ii in xx:
                array_comp += (array_comp >= ii) * 1
            composed_indices = np.concatenate((xx, array_comp))

        return composed_indices

    def update_lpi(self, lpi, removed_inds):
        # update lpi with removed indices from rpi
        counts = np.array([sum(1 for num in removed_inds if num < x) for x in lpi])
        lpi = np.array(lpi) - counts
        return lpi

    def update_rpi(self, rpi, removed_inds):
        # update lpi with removed indices from rpi
        counts = np.array([sum(1 for num in removed_inds if num < xx) for xx in rpi])
        rpi = np.array(rpi) - counts
        return rpi
