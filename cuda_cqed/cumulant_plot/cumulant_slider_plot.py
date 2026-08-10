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

COLORS = [(1,0,0),(0,0,1),(0.2,1,0.2), (0.5,0.5,0.2)]

def cumulant_slider_plot(t, mode_names, a_list, aa_list, na_list,
                 axes_dict: dict, plot_range=5, callback: Callable = None, adaptiveRange=False,
                 **hist2dArgs) -> List[Slider]:
    """Create a slider plot widget. The caller needs to maintain a reference to
    the returned Slider objects to keep the widget activate

    This is designed for plotting the Q fucntion of states represented by second order cumulant expansion
    thus you need to provide the operators a, aa, and adaga or na

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
    num_axes = len(axes_dict)

    try:
        num_modes = len(a_list)
        num_states = len(a_list[0])
    except:
        num_modes = 1
        a_list = [[a_list]]
        aa_list = [[aa_list]]
        na_list = [[na_list]]

    fig, axs = plt.subplots(nrows=2, ncols=num_modes, figsize=(9,9 + num_axes * 0.3))
    callback_text = plt.figtext(0.15, 0.01, "", size="large", figure=fig)
    plt.subplots_adjust(bottom=num_axes * 0.3 / (9 + num_axes * 0.3) + 0.1)

    for i in range(0, num_modes):
        axs[0, i].set_xlim([-plot_range, plot_range])
        axs[0, i].set_ylim([-plot_range, plot_range])
        axs[0, i].grid()
        axs[0, i].set_title(str(mode_names[i]))
        axs[1, i].grid()
        axs[0, i].set_aspect(1)
        axs[0, i].set_yticks(axs[0, i].get_xticks())

        axs[1, i].set_xlabel('Time (ns)')
        axs[1, i].set_ylabel('Mean amplitude')


    blob_lines = [[0]*num_states for i in range(num_modes)]
    abs_lines = [[0]*num_states for i in range(num_modes)]
    real_lines = [[0] * num_states for i in range(num_modes)]
    imag_lines = [[0] * num_states for i in range(num_modes)]
    rms_lines = [[0] * num_states for i in range(num_modes)]
    time_scatter = [[0]*num_states for i in range(num_modes)]

    for i in range(0, num_modes):
        ax_ylim = np.max(np.abs(a_list[i]))*1.1
        axs[1, i].set_ylim([-ax_ylim, ax_ylim])
        for j in range(0, num_states):
            blob_lines[i][j] = plot_Qfunc_outline(a_list[i][j].flatten()[0], aa_list[i][j].flatten()[0], na_list[i][j].flatten()[0], ax=axs[0, i], color=COLORS[j], polygon=None)

            # abs_lines[i][j] = axs[1, i].plot(t, np.abs(a_list[i][j].flatten()[0:len(t)]), c=COLORS[j], alpha=0.5)
            real_lines[i][j] = axs[1, i].plot(t*1e9, np.real(a_list[i][j].flatten()[0:len(t)]), c=COLORS[j], alpha=0.5)
            imag_lines[i][j] = axs[1, i].plot(t*1e9, np.imag(a_list[i][j].flatten()[0:len(t)]), dashes=[2, 2, 10, 2], c=COLORS[j], alpha=0.5)
            # rms_lines[i][j] = axs[1, i].plot(t, np.sqrt(na_list[i][j].flatten()[0:len(t)]), linewidth=1, c=COLORS[j], alpha=0.5)
            time_scatter[i][j] = axs[1, i].plot([0], [0], 'ko')

    axcolor = 'lightgoldenrodyellow'
    sld_list = []
    for idx, (k, v) in enumerate(axes_dict.items()):
        ax_ = plt.axes([0.2, (num_axes - idx) * 0.04, 0.6, 0.03], facecolor=axcolor)
        sld_ = Slider(ax_, k, 0, len(v) - 1, valinit=0, valstep=1)
        sld_list.append(sld_)

    # update funtion
    def update(val):
        sel_dim = []
        ax_val_list = []
        ax_idx_list = []
        for i in range(num_axes):
            ax_name = sld_list[i].label.get_text()
            ax_idx = int(sld_list[i].val)
            sel_dim.append(int(ax_idx))
            ax_val = np.round(axes_dict[ax_name][ax_idx], 5)
            ax_val_list.append(ax_val)
            ax_idx_list.append(ax_idx)
            sld_list[i].valtext.set_text(str(ax_val))
        ax_idx_tuple = tuple(ax_idx_list)

        for i in range(0, num_modes):
            for j in range(0, num_states):
                new_a = a_list[i][j][ax_idx_tuple]
                new_aa = aa_list[i][j][ax_idx_tuple]
                new_na = na_list[i][j][ax_idx_tuple]

                line = plot_Qfunc_outline(new_a, new_aa, new_na, ax=axs[0, i], polygon=blob_lines[i][j])
                # abs_lines[i][j][0].set_ydata(np.abs(a_list[i][j][ax_idx_tuple[0:-1]]))
                real_lines[i][j][0].set_ydata(np.real(a_list[i][j][ax_idx_tuple[0:-1]]))
                imag_lines[i][j][0].set_ydata(np.imag(a_list[i][j][ax_idx_tuple[0:-1]]))
                # rms_lines[i][j][0].set_ydata(np.sqrt(na_list[i][j][ax_idx_tuple[0:-1]]))

                time_scatter[i][j][0].set_xdata([t[ax_idx_tuple[-1]]*1e9])
                # time_scatter[i][j][0].set_ydata([np.abs(a_list[i][j][ax_idx_tuple])])

            # print callback result on top of figure
                if callback is not None:
                    result = callback(new_a, new_aa, new_na, *ax_val_list)
                    callback_text.set_text(callback.__name__ + f": {result}")
                fig.canvas.draw_idle()

    for i in range(num_axes):
        sld_list[i].on_changed(update)
    return sld_list

def plot_Qfunc_outline(a, aa, na, ax=None, color=(1,0,0), polygon=None):
    if ax == None:
        ax = plt.gca()

    ad = np.conjugate(a)
    adad = np.conjugate(aa)
    sxx = 1 / 2 + 1 / 2 * ((aa - a * a) + 2 * (na - ad * a) + (adad - ad * ad))
    syy = 1 / 2 - 1 / 2 * ((aa - a * a) - 2 * (na - ad * a) + (adad - ad * ad))
    sxy = np.imag(1 / 2 * ((aa - a * a) - (adad - ad * ad)))
    covar = np.real(np.array([[sxx, sxy], [sxy, syy]]))

    phi = np.linspace(0, 2 * np.pi, 101)

    theta = np.angle(np.linalg.eig(covar)[1][0][0] + 1j * np.linalg.eig(covar)[1][0][1])
    Sx = np.linalg.eig(covar)[0][0]
    Sy = np.linalg.eig(covar)[0][1]

    x1 = (Sx) ** (1 / 4) * np.cos(phi) * 2
    y1 = (Sy) ** (1 / 4) * np.sin(phi) * 2

    x2 = x1 * np.cos(theta) + y1 * np.sin(theta) + np.real(a)
    y2 = -x1 * np.sin(theta) + y1 * np.cos(theta) + np.imag(a)

    if polygon == None:
        # line = ax.plot(x2, y2)[0]
        polygon = ax.fill(x2, y2, color=color, alpha=0.5)[0]
    else:
        polygon.set_xy(np.array([x2, y2]).transpose())
        # line.set_xdata(x2)
        # line.set_ydata(y2)
        plt.draw()

    # return line
    return polygon
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

    sim = Sim(use_complex=True)

    pulse_length = 50e-9
    ramp = pulse_length * 0.5
    gap = ramp * 5

    sim.add_param('hi', 0, is_excitation=True)
    sim.add_paramsweep('sigma_z', -1, 1, 2)
    sim.add_paramsweep('chi', 0e6 * 2 * pi, 2e6 * 2 * pi, 5)
    sim.add_param('g_3', 5e6 * 2 * np.pi)
    sim.add_param('K4', 0)
    sim.add_param('K6', 0.0e6 * 2 * np.pi)
    sim.add_param('lambda_ab', 0.1)
    sim.add_param('lambda_bc', 0.1)
    sim.add_param('ka', 0.0e6 * 2 * np.pi)
    sim.add_param('gamma_phi_b', 0.0e6 * 2 * np.pi)
    sim.add_paramsweep('kb', 0.0e6 * 2 * np.pi, 10e6 * 2 * np.pi, 10)
    sim.add_param('kc', 5e6 * 2 * np.pi)
    sim.add_param('A', 0)

    sim.add_param('length', pulse_length)
    sim.add_param('ramp', ramp)

    sim.add_param('wCONV1', 0)
    sim.add_param('amplCONV1', 1.7)
    sim.add_param('phaseCONV1', np.pi / 2)

    sim.add_param('wSQZ', 0.2)
    sim.add_param('amplSQZ', 0.2*0)
    sim.add_param('phaseSQZ', np.pi / 2)

    sim.add_param('wCONV2', 0)
    sim.add_param('amplCONV2', 2.2*0)
    sim.add_param('phaseCONV2', -np.pi / 2)

    sim.add_param('init_a', 3)
    sim.add_param('init_b', 0)
    sim.add_param('init_c', 0)

    sim.add_paramsweep('delay', 100e-9, 300e-9, 5)

    pulse_sequence_1 = '0'
    pulse_sequence_2 = '0'
    pulse_sequence_3 = '0'

    tpulse = 0

    pulse = sim.make_pulse('wCONV1', 'amplCONV1', 'phaseCONV1', str(tpulse)+'+delay', str(tpulse + pulse_length)+'+delay', str(ramp))
    pulse_sequence_1 = sim.make_pulse_sequence([pulse_sequence_1, pulse])
    tpulse += pulse_length + gap

    pulse = sim.make_pulse('wSQZ', 'amplSQZ', 'phaseSQZ', str(tpulse)+'+delay', str(tpulse+pulse_length)+'+delay', str(ramp))
    pulse_sequence_2 = sim.make_pulse_sequence([pulse_sequence_2, pulse])
    tpulse += pulse_length+gap

    pulse = sim.make_pulse('wCONV2', 'amplCONV2', 'phaseCONV2', str(tpulse)+'+delay', str(tpulse+pulse_length)+'+delay', str(ramp))
    pulse_sequence_3 = sim.make_pulse_sequence([pulse_sequence_3, pulse])
    tpulse += pulse_length+gap

    sim.add_EOM('eta_1', pulse_sequence_1)
    sim.add_EOM('eta_2', pulse_sequence_2)
    sim.add_EOM('eta_3', pulse_sequence_3)

    sim.add_EOM('a', ' (-1.0*1j*chi*sigma_z)*a +(-6.0*1j*conjugate(eta_1)*g_3*lambda_ab)*b  -ka*a/2', IC_str='init_a')
    sim.add_EOM('aa', ' (-2.0*1j*chi*sigma_z)*aa +(-12.0*1j*conjugate(eta_1)*g_3*lambda_ab)*ab  -ka*aa', IC_str='init_a**2')
    sim.add_EOM('adaga',
                ' (6.0*1j*eta_1*g_3*lambda_ab)*conjugate(adagb) +(-6.0*1j*conjugate(eta_1)*g_3*lambda_ab)*adagb  -ka*adaga',
                IC_str='init_a**2')
    sim.add_EOM('b',
                ' (-6.0*1j*eta_1*g_3*lambda_ab)*a +(1.0*1j*(-2.0*K4*(abs(eta_1)**2+abs(eta_2)**2+abs(eta_3)**2) - 1.0*K6*(abs(eta_1)**2+abs(eta_2)**2+abs(eta_3)**2)**2))*b +(-6.0*1j*eta_2*g_3)*conjugate(b) +(-2.0*1j*K4 )*bdagb*b +(-6.0*1j*conjugate(eta_3)*g_3*lambda_bc)*c  -kb*b/2-gamma_phi_b*b',
                IC_str='init_b')
    sim.add_EOM('bb',
                ' (-6.0*1j*eta_2*g_3)+(-12.0*1j*eta_1*g_3*lambda_ab)*ab +(-2.0*1j*K4 + 2.0*1j*(-2.0*K4*(abs(eta_1)**2+abs(eta_2)**2+abs(eta_3)**2) - 1.0*K6*(abs(eta_1)**2+abs(eta_2)**2+abs(eta_3)**2)**2))*bb +(-12.0*1j*conjugate(eta_3)*g_3*lambda_bc)*bc +(-12.0*1j*eta_2*g_3)*bdagb +(-4.0*1j*K4 )*bdagb*bb  -kb*bb-2*gamma_phi_b*bb',
                IC_str='init_b**2')
    sim.add_EOM('bdagb',
                ' (-6.0*1j*eta_1*g_3*lambda_ab)*conjugate(adagb) +(6.0*1j*conjugate(eta_1)*g_3*lambda_ab)*adagb +(6.0*1j*conjugate(eta_2)*g_3)*bb +(6.0*1j*eta_3*g_3*lambda_bc)*conjugate(bdagc) +(-6.0*1j*eta_2*g_3)*conjugate(bb) +(-6.0*1j*conjugate(eta_3)*g_3*lambda_bc)*bdagc  -kb*bdagb',
                IC_str='init_b**2')
    sim.add_EOM('c', ' (-6.0*1j*eta_3*g_3*lambda_bc)*b  -kc*c/2', IC_str='init_c')
    sim.add_EOM('cc', ' (-12.0*1j*eta_3*g_3*lambda_bc)*bc  -kc*cc', IC_str='init_c**2')
    sim.add_EOM('cdagc',
                ' (-6.0*1j*eta_3*g_3*lambda_bc)*conjugate(bdagc) +(6.0*1j*conjugate(eta_3)*g_3*lambda_bc)*bdagc  -kc*cdagc',
                IC_str='init_c**2')
    sim.add_EOM('ab',
                ' (-6.0*1j*eta_1*g_3*lambda_ab)*aa +(-1.0*1j*chi*sigma_z + 1.0*1j*(-2.0*K4*(abs(eta_1)**2+abs(eta_2)**2+abs(eta_3)**2) - 1.0*K6*(abs(eta_1)**2+abs(eta_2)**2+abs(eta_3)**2)**2))*ab +(-6.0*1j*eta_2*g_3)*conjugate(adagb) +(-2.0*1j*K4 )*conjugate(adagb)*bb +(-6.0*1j*conjugate(eta_3)*g_3*lambda_bc)*ac +(-6.0*1j*conjugate(eta_1)*g_3*lambda_ab)*bb  -(ka+kb)*ab/2')
    sim.add_EOM('adagb',
                ' (-6.0*1j*eta_1*g_3*lambda_ab)*adaga +(1.0*1j*chi*sigma_z + 1.0*1j*(-2.0*K4*(abs(eta_1)**2+abs(eta_2)**2+abs(eta_3)**2) - 1.0*K6*(abs(eta_1)**2+abs(eta_2)**2+abs(eta_3)**2)**2))*adagb +(-6.0*1j*eta_2*g_3)*conjugate(ab) +(-2.0*1j*K4 )*conjugate(ab)*bb +(-6.0*1j*conjugate(eta_3)*g_3*lambda_bc)*adagc +(6.0*1j*eta_1*g_3*lambda_ab)*bdagb  -(ka+kb)*adagb/2')
    sim.add_EOM('bc',
                ' (-6.0*1j*eta_1*g_3*lambda_ab)*ac +(-6.0*1j*eta_3*g_3*lambda_bc)*bb +(1.0*1j*(-2.0*K4*(abs(eta_1)**2+abs(eta_2)**2+abs(eta_3)**2) - 1.0*K6*(abs(eta_1)**2+abs(eta_2)**2+abs(eta_3)**2)**2))*bc +(-2.0*1j*K4 )*conjugate(bdagb)*bc +(-6.0*1j*eta_2*g_3)*bdagc +(-6.0*1j*conjugate(eta_3)*g_3*lambda_bc)*cc  -(kb+kc)*bc/2')
    sim.add_EOM('bdagc',
                ' (6.0*1j*conjugate(eta_1)*g_3*lambda_ab)*adagc +(6.0*1j*conjugate(eta_2)*g_3)*bc +(-6.0*1j*eta_3*g_3*lambda_bc)*bdagb +(2.0*1j*K4 + 1.0*1j*(-2.0*K4*(abs(eta_1)**2+abs(eta_2)**2+abs(eta_3)**2) - 1.0*K6*(abs(eta_1)**2+abs(eta_2)**2+abs(eta_3)**2)**2) + 1.0*1j*(2.0*K4*(abs(eta_1)**2+abs(eta_2)**2+abs(eta_3)**2) + 1.0*K6*(abs(eta_1)**2+abs(eta_2)**2+abs(eta_3)**2)**2) )*conjugate(bb)*bc +(1.0*1j*(2.0*K4*(abs(eta_1)**2+abs(eta_2)**2+abs(eta_3)**2) + 1.0*K6*(abs(eta_1)**2+abs(eta_2)**2+abs(eta_3)**2)**2))*bdagc +(6.0*1j*eta_3*g_3*lambda_bc)*cdagc  -(kb+kc)*bdagc/2')
    sim.add_EOM('ac',
                ' (-6.0*1j*eta_3*g_3*lambda_bc)*ab +(-1.0*1j*chi*sigma_z)*ac +(-6.0*1j*conjugate(eta_1)*g_3*lambda_ab)*bc  -(ka+kc)*ac/2')
    sim.add_EOM('adagc',
                ' (-6.0*1j*eta_3*g_3*lambda_bc)*adagb +(1.0*1j*chi*sigma_z)*adagc +(6.0*1j*eta_1*g_3*lambda_ab)*bdagc  -(ka+kc)*adagc/2')

    sim.set_solve_type('all')

    sim.specify_time(t_f=1e-6, pts=1001)

    sim.validate()

    x, t = sim.solve()

    a = x[6, :] + 1j * x[7, :]
    aa = x[8, :] + 1j * x[9, :]
    na = x[10, :]

    b = x[12, :] + 1j * x[13, :]
    bb = x[14, :] + 1j * x[15, :]
    nb = x[16, :]

    c = x[18, :] + 1j * x[19, :]
    cc = x[20, :] + 1j * x[21, :]
    nc = x[22, :]

    a0 = a[0, :, :, :, :]
    a1 = a[1, :, :, :, :]
    # a2 = a[2, :, :, :]
    aa0 = aa[0, :, :, :, :]
    aa1 = aa[1, :, :, :, :]
    # aa2 = aa[2, :, :, :]
    na0 = na[0, :, :, :, :]
    na1 = na[1, :, :, :, :]
    # na2 = na[2, :, :, :]

    b0 = b[0, :, :, :, :]
    b1 = b[1, :, :, :, :]
    # b2 = b[2, :, :, :]
    bb0 = bb[0, :, :, :, :]
    bb1 = bb[1, :, :, :, :]
    # bb2 = bb[2, :, :, :]
    nb0 = nb[0, :, :, :, :]
    nb1 = nb[1, :, :, :, :]
    # nb2 = nb[2, :, :, :]

    c0 = c[0, :, :, :, :]
    c1 = c[1, :, :, :, :]
    # c2 = c[2, :, :, :]
    cc0 = cc[0, :, :, :, :]
    cc1 = cc[1, :, :, :, :]
    # cc2 = cc[2, :, :, :]
    nc0 = nc[0, :, :, :, :]
    nc1 = nc[1, :, :, :, :]
    # nc2 = nc[2, :, :, :]

    axes_dict = {'chi (kHz)': sim.paramsweep_dict['chi']/(2e3*np.pi), 'kb (MHz)': sim.paramsweep_dict['kb']/(2e6*np.pi), 'delay (ns)': sim.paramsweep_dict['delay']*(1e9),
                 'time (ns)': np.unique(t) * 1e9}

    plt.close('all')
    # cumulant_slider_plot([[a0, b0, c0],[a1, b1, c1]], [[aa0, bb0, cc0],[aa1, bb1, cc1]], [[na0, nb0, nc0],[na1, nb1, nc1]], axes_dict, plot_range=20)
    cumulant_slider_plot(np.unique(t), ['Readout','Amplifier','Output'], [[a0, a1], [b0, b1], [c0, c1]], [[aa0, aa1], [bb0, bb1], [cc0, cc1]],
                         [[na0, na1], [nb0, nb1], [nc0, nc1]], axes_dict, plot_range=20)

    # cumulant_slider_plot(np.unique(t), ['Readout','Amplifier','Output'], [[a0, a1, a2], [b0, b1, b2], [c0, c1, c2]],
    #                      [[aa0, aa1, aa2], [bb0, bb1, bb2], [cc0, cc1, cc2]],
    #                      [[na0, na1, na2], [nb0, nb1, nb2], [nc0, nc1, nc2]], axes_dict, plot_range=20)