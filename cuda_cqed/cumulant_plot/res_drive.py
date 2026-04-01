# -*- coding: utf-8 -*-
"""
Created Jan 26 2026

@author: chao, boris
"""
import itertools

from typing import Union, List, Callable, Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import h5py
from matplotlib.animation import FuncAnimation
import warnings
import matplotlib as mpl

mpl.use('Qt5Agg')


def cumulant_slider_plot(a, aa, na,
                         axes_dict: dict, plot_range=5, callback: Callable = None, adaptiveRange=False,
                         **hist2dArgs) -> List[Slider]:
    """Create a slider plot widget. The caller needs to maintain a reference to
    the returned Slider objects to keep the widget activate

    :param data_I:
    :param data_Q:
    :param axes_dict: a dictionary that contains the data of each axis
    :param hist2dArgs:
    :return: list of Slider objects.
    """
    try:  # incase it's a datadict
        axes_dict.to_dict()
    except AttributeError:
        pass

    # initial figure
    nAxes = len(axes_dict)

    fig = plt.figure(figsize=(7, 7 + nAxes * 0.3))
    callback_text = plt.figtext(0.15, 0.01, "", size="large", figure=fig)
    plt.subplots_adjust(bottom=nAxes * 0.3 / (7 + nAxes * 0.3) + 0.1)
    plt.subplot(1, 1, 1)
    main_ax = plt.gca()
    main_ax.set_xlim([-plot_range, plot_range])
    main_ax.set_ylim([-plot_range, plot_range])
    main_ax.grid()

    line = plot_Qfunc(a[0, 0], aa[0, 0], na[0, 0], ax=None, line=None)

    axcolor = 'lightgoldenrodyellow'
    sld_list = []
    for idx, (k, v) in enumerate(axes_dict.items()):
        ax_ = plt.axes([0.2, (nAxes - idx) * 0.04, 0.6, 0.03], facecolor=axcolor)
        sld_ = Slider(ax_, k, 0, len(v) - 1, valinit=0, valstep=1)
        sld_list.append(sld_)

    # update funtion
    def update(val):
        sel_dim = []
        ax_val_list = []
        ax_idx_list = []
        for i in range(nAxes):
            ax_name = sld_list[i].label.get_text()
            ax_idx = int(sld_list[i].val)
            sel_dim.append(int(ax_idx))
            ax_val = np.round(axes_dict[ax_name][ax_idx], 5)
            ax_val_list.append(ax_val)
            ax_idx_list.append(ax_idx)
            sld_list[i].valtext.set_text(str(ax_val))
        ax_idx_tuple = tuple(ax_idx_list)
        new_a = a[ax_idx_tuple]
        new_aa = aa[ax_idx_tuple]
        new_na = na[ax_idx_tuple]

        line = plot_Qfunc(new_a, new_aa, new_na, ax=main_ax, line=main_ax.lines[0])
        # print callback result on top of figure
        if callback is not None:
            result = callback(new_a, new_aa, new_na, *ax_val_list)
            callback_text.set_text(callback.__name__ + f": {result}")
        fig.canvas.draw_idle()

    for i in range(nAxes):
        sld_list[i].on_changed(update)
    return sld_list


def plot_Qfunc(a, aa, na, ax=None, line=None):
    if ax == None:
        ax = plt.gca()

    ad = np.conjugate(a)
    adad = np.conjugate(aa)
    sxx = 1 / 2 + 1 / 2 * ((aa - a * a) + 2 * (na - ad * a) + (adad - ad * ad))
    syy = 1 / 2 - 1 / 2 * ((aa - a * a) - 2 * (na - ad * a) + (adad - ad * ad))
    sxy = np.imag(1 / 2 * ((aa - a * a) - (adad - ad * ad)))
    covar = np.real(np.array([[sxx, sxy], [sxy, syy]]))

    phi = np.linspace(0, 2 * np.pi, 101)

    order = np.argsort(np.linalg.eig(covar)[0])
    major_idx = order[1]
    minor_idx = order[0]

    theta = -np.angle(np.linalg.eig(covar)[1][:, 1][minor_idx] + 1j * np.linalg.eig(covar)[1][:, 1][major_idx])
    Smajor = np.sort(np.linalg.eig(covar)[0])[major_idx]
    Sminor = np.sort(np.linalg.eig(covar)[0])[minor_idx]

    x1 = (Smajor) ** (1 / 4) * np.cos(phi) * 2
    y1 = (Sminor) ** (1 / 4) * np.sin(phi) * 2

    x2 = x1 * np.cos(theta) + y1 * np.sin(theta) + np.real(a)
    y2 = -x1 * np.sin(theta) + y1 * np.cos(theta) + np.imag(a)

    if line == None:
        line = ax.plot(x2, y2)
    else:
        line.set_xdata(x2)
        line.set_ydata(y2)
        plt.draw()

    return line
    # ax.xlim([-7,7])
    # ax.ylim([-7,7])
    # plt.gca().set_aspect('equal')
    # plt.grid()
    # plt.show()


#
# def AnimatePColorMesh(xdata, ydata, zdata,
#                       axes_dict: dict, fileName="", **pColorMeshArgs):
#     try:  # incase it's a datadict
#         axes_dict.to_dict()
#     except AttributeError:
#         pass
#
#     if len(axes_dict.keys()) > 1:
#         raise NotImplementedError("this function (axis > 1) is still under developing")
#     pColorMeshArgs["shading"] = pColorMeshArgs.get("shading", "auto")
#     pColorMeshArgs["vmin"] = pColorMeshArgs.get("vmin", np.min(zdata))
#     pColorMeshArgs["vmax"] = pColorMeshArgs.get("vmax", np.max(zdata))
#     # initial figure
#     nAxes = len(axes_dict)
#     zdata0 = _indexData(zdata, np.zeros(nAxes))
#     fig = plt.figure(figsize=(7, 7 + nAxes * 0.3))
#
#     callback_text = plt.figtext(0.15, 0.01, "", size="large", figure=fig)
#     plt.subplots_adjust(bottom=nAxes * 0.3 / (7 + nAxes * 0.3) + 0.1)
#     plt.subplot(1, 1, 1)
#     pcm = plt.pcolormesh(xdata, ydata, zdata0.T, **pColorMeshArgs)
#     ax1 = plt.gca()
#     fig.colorbar(pcm, ax=ax1)
#     axcolor = 'lightgoldenrodyellow'
#     for k, v in axes_dict.items():
#         sweepLabel = k
#         sweepValue = v
#
#     # update funtion
#     def update(val):
#         sel_dim = val
#         newZdata = _indexData(zdata, [sel_dim])
#         ax1.cla()
#         pcm = ax1.pcolormesh(xdata, ydata, newZdata.T, **pColorMeshArgs)
#         ax1.set_title(sweepLabel + ": " + str(sweepValue[val]))
#         fig.canvas.draw_idle()
#
#     anim = FuncAnimation(fig, update, frames=np.arange(len(sweepValue)), interval=500)
#     if fileName != "":
#         anim.save(fileName + ".gif", dpi=80, writer='imagemagick')
#     return anim


if __name__ == '__main__':
    from cuda_cqed.sim import Sim
    # import gpu_odes.HatGPUODE_D
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib

    # matplotlib.use('Qt5Agg')

    pi = np.pi
    K = 0.0001
    g3 = 4e-3
    s1 = 1
    g2 = g3 * s1

    sim = Sim(use_complex=True)

    sim.add_paramsweep('wd', -5 * 2 * pi, 5 * 2 * pi, 101, is_excitation=True)
    sim.add_param('wb', 0 * 2 * pi)
    sim.add_param('sqrtkb', np.sqrt(5 * 2 * np.pi))  # in MHz
    sim.add_param('g3', g3)
    sim.add_param('K', K)
    sim.add_param('A', 1)

    sim.add_drive_EOM('bin', 'A*exp(-1j*t*wd)')
    sim.add_EOM('b', '-1j*wb*b - bin*sqrtkb**2 - (sqrtkb)/2*b')
    sim.set_solve_type('all')

    sim.specify_time(t_f=100, pts=10001)

    sim.validate(print_result=True)

    # x, t = sim.solve()
    #
    # xd = x.copy()
    # td = t.copy()
    #
    # b = x[0, :] + 1j * x[1, :]
    # bin = x[2, :] + 1j * x[3, :] # drive terms are always last
    #
    # plt.figure()
    # plt.plot(np.abs(b[:, -1].transpose()))
    #
    # plt.figure()
    # plt.plot(np.real(bin[49,:]))
    #
    x, t = sim.quick_trace()

    # plt.figure(1)
    # plt.clf()
    # plt.plot(t * 1e9, x[0, :] / np.max(x[0, :]), color=(1, 0, 0, 0.7), label='a in')
    # plt.plot(t * 1e9, x[2, :] / np.max(x[2, :]) + 2, color=(0, 1, 0, 0.5), label='a')
    # plt.xlabel('Time (ns)')
    # plt.ylabel('Normalized amplitude')
    # plt.legend()
    # plt.grid()
    # plt.show()

