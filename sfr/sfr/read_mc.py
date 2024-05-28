import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import numpy as np
import warnings
from sfrec import find_decomposition_clear, Decomposition
from const import *
from tools import norm_to_unit_std
from plot_tools import plot_gram_matrix, plot_subplots, annotatep

titleopts = {"loc": "left","position":(.0,1.00)}

def settitle(handle, number, title, opts=titleopts):
    handle.text(-.13,1.05, '({})'.format(chr(number+97)),
            transform=handle.transAxes,fontsize=22)
    handle.set_title(title, **opts)

def add_caption(fighandle, text, x=0.02, y=0.02, opts={}):
    fighandle.text( x, y, text, wrap=True, **opts)

def analysis(
    rr, bindex=None, b_singlefig=1, b_saveplots=False, b_showplots=True, **kwargs
):

    if np.all([hasattr(rr, key) for key in ["spatial_coherence", "nmse", "id"]]) & (
        np.sum([b_saveplots, b_showplots]) == 0
    ):
        if rr.id == rr.title():
            return
        else:
            [delattr(rr, key) for key in ["spatial_coherence", "nmse", "id"]]
            return analysis(rr, bindex, b_singlefig, b_saveplots, b_showplots, **kwargs)
    else:
        mf = rr.measured_field
        rf = rr.reconstructed_field
        fstamp = rr.fit_timestamp

        ### De-compress
        # future also re-read measurements, only store random variables
        if not hasattr(mf, "_pm"): mf._gen_field()  # re-read

        ### input, reconstruction, reference, error
        # n_points in relation to nyq sampling along one dimension
        mf.n_points = np.prod(mf.sidx.shape)
        spacing_in_nyq = (
            mf.n_points
            / np.ceil((mf.dx * (mf.shp[0] - 1)) / (c0 / mf.frequency / 2)) ** 2
        )

        rr.assess_reconstruction()

        # PLOTS
        if b_showplots + b_saveplots:
            items = rr.transform_opts.items()
            if bindex:
                titlestring = "".join(
                    [
                        str(bindex),
                        " ",
                        str(bin(bindex).replace("0b", "")),
                        "\n",
                        rr.decomposition,
                        " ",
                        rr.title(),
                        "\n",
                        mf.measurement,
                        " @ {:.0f} Hz, ".format(mf.frequency),
                        mf.loss_mode,
                        ": ",
                        str(mf.spatial_sampling),
                        "%]\n",
                        "Transform ",
                        rr.transform,
                        " ",
                        str(sorted(items))
                        .replace("'", "")
                        .replace("reg_", "")
                        .replace(",", ""),
                    ]
                )
            else:
                titlestring = "".join(
                    [
                        rr.decomposition,
                        " ",
                        rr.title(),
                        "\n",
                        mf.measurement,
                        " @ {:.0f} Hz, ".format(mf.frequency),
                        mf.loss_mode,
                        ": ",
                        str(mf.spatial_sampling),
                        "\n",
                        "Transform ",
                        rr.transform,
                        " ",
                        str(sorted(items))
                        .replace("'", "")
                        .replace("reg_", "")
                        .replace(",", ""),
                    ]
                )

            # FIG 1
            if b_singlefig == 2 :
                fig = plt.figure(1, figsize=(15, 3.5))
                plt.clf()
                # gs1 = gs.GridSpec(2, 2, figure=fig, 
                        # top=0.85, bottom=0.1,
                        # hspace = .4)
                gs1 = gs.GridSpec(1, 4, figure=fig, 
                        top=0.80,
                        right= .93,
                        wspace = .9)
            elif b_singlefig:
                fig = plt.figure(1, figsize=(10, 5))
                plt.clf()
                gs0 = gs.GridSpec(1, 1, figure=fig, hspace=0.15, top=0.92, bottom=0.20)
                gs1 = gs.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs0[0], wspace=0.7)
            else:
                fig = plt.figure(1, figsize=(9.42, 3.33))
                plt.clf()
                gs1 = gs.GridSpec(1, 4, figure=fig, wspace=0.40, hspace=0.3)

            axes = [
                fig.add_subplot(gs1[0]),
                fig.add_subplot(gs1[1]),
                fig.add_subplot(gs1[2]),
                fig.add_subplot(gs1[3]),
            ]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                norm = 10 * np.log10(np.mean(np.abs(mf.p)**2)),
                plot_subplots(
                    axes,
                    [
                        20 * np.log10(np.abs(mf.sp))-norm,
                        20 * np.log10(np.abs(rf.p))-norm,
                        20 * np.log10(np.abs(mf.p))-norm,
                        20 * np.log10(np.abs(mf.p - rf.p) / np.abs(mf.p)),
                    ],
                    ["a", "b", "c", "d", "e", "f", "g"],
                    dims=np.array(
                        [
                            [np.min(mf.r[:, 0]), np.max(mf.r[:, 0])],
                            [np.min(mf.r[:, 1]), np.max(mf.r[:, 1])],
                            [np.min(mf.r[:, 2]), np.max(mf.r[:, 2])],
                        ]
                    ),
                    vrange=[-20, 10],
                )
            if b_singlefig == 2:
                declabel = "".join(["$_{",find_decomposition_clear(rr.decomposition),"}$"])
                eloc = {'x':1.30, 'y':1.08} # left of title
                eaxes = axes[3]
            else:
                declabel = rr.tag
                eloc = {'x':1.15, 'y':1.30}
                eaxes = axes[3]
            settitle(axes[0], 0, r"$\mathbf{p}_{m}$")
            settitle(axes[1], 1, r"$\mathbf{\hat{p}}$" + declabel)
            settitle(axes[2], 2, r"$\mathbf{p}_{reference}$")
            titleopts.update({'usetex':False})
            settitle(axes[3], 3, r"$\varepsilon$"+ declabel) 
            _= titleopts.pop('usetex')
            if b_singlefig == 2:
                annotatep( axes[1], r"$\gamma$"+" {:.2f}".format(
                    rr.spatial_coherence), **eloc, usetex = False,)
                annotatep( axes[3], "NMSE" + " {:.2f} dB".format(
                    rr.nmse), **eloc, usetex = False,)
                annotatep( axes[2], "Cardinality \u2300" + " {:.2f}".format(
                    rr.avg_nonzeros), **eloc, usetex = False,)
            else:
                annotatep(
                    eaxes,
                    "NMSE" + " = {:.2f} dB".format(error),
                    + "\n"
                    + r"$\gamma$"+" = {:.2f}".format(rr.spatial_coherence)
                    # + "$< \epsilon _n ^2 >$"
                    **eloc, usetex = False,
                )
            if not b_singlefig:
                if b_saveplots:
                    plt.savefig(rr.title() + "_rec.pdf")
                    plt.close()
                if b_showplots:
                    plt.draw()
                else:
                    plt.close()

            ### FIG 2 reconstruction pressure level and coefficient statistics
            if False:
                E = np.mean(np.abs(mf.p) ** 2)
                L0 = 10 * np.log10(E)
                if not b_singlefig:
                    fig = plt.figure(2, figsize=(9.42, 3.33))
                    plt.clf()
                    gs2 = gs.GridSpec(
                        1, 4, figure=fig, wspace=0.3, hspace=0.4, bottom=0.10, top=0.85
                    )
                axes = [
                    fig.add_subplot(gs2[:, 0]),
                    fig.add_subplot(gs2[:, 1]),
                    fig.add_subplot(gs2[0, 2]),
                    fig.add_subplot(gs2[1, 2]),
                    fig.add_subplot(gs2[:, 3]),
                ]
                l1 = np.log(10) / 10
                x = np.linspace(0 - 40, 0 + 10, 100)
                poisson = l1 * np.exp(l1 * (x - 0) - np.exp(l1 * (x - 0)))
                # consider statsmodels.api.qqplot_2samples(x,y)
                axes[0].plot(x, poisson, "k", alpha=0.6, label="$\sum ref$")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    axes[0].hist(
                        20 * np.log10(np.abs(norm_to_unit_std(mf.fp))).ravel() - 6,
                        color="C3",
                        bins=50,
                        density=True,
                        alpha=0.9,
                        label="true",
                    )
                    axes[0].hist(
                        20
                        * np.log10(np.abs(norm_to_unit_std(mf.fp)[mf.sidx])).ravel()
                        - 6,
                        color="C2",
                        bins=50,
                        density=True,
                        alpha=0.4,
                        label="samples",
                    )
                    tmp = 20 * np.log10(np.abs(norm_to_unit_std(rf.p))).ravel() - 6
                try:
                    axes[0].hist(
                        tmp,
                        bins=50,
                        range=(
                            np.floor(np.max([np.min(tmp), -60])),
                            np.ceil(np.max(tmp)),
                        ),
                        density=True,
                        alpha=0.5,
                        label=rr.tag,
                    )
                except:
                    pass
                axes[0].legend(ncol=1, loc="upper left", fontsize='medium')
                settitle(axes[0], 4, "[dB rel $<p^2>$]")
                popt = {"bins": 30, "alpha": 0.7, "density": True, "log": True}
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    if not np.any(rr.gammar):
                        rr.gammar = np.empty(1)
                    elif np.all(np.isnan(rr.gammar)):
                        rr.gammar = np.array([0])
                    axes[1].hist(
                        np.abs(np.angle(rr.gammar.T).ravel() / np.pi % (1 / 2)),
                        label=rr.tag,
                        **popt
                    )
                settitle(axes[1], 5, r"phase $\gamma$ [pi % 1/2]")
                axes[1].legend(ncol=1, loc="upper right", fontsize='medium')
                # axes[1].set_yticks([])
                popt.update({"log": False})
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    axes[2].hist(
                        np.abs(rr.gammar.T).ravel(), color="C0", label=rr.tag, **popt
                    )
                settitle(axes[2], 6, r"hist |$\gamma$|")
                axes[2].legend(ncol=1, loc="upper right", fontsize='medium')
                ax = plot_gram_matrix(rr.Ar, axes[4], rr.tag, number=8)
                # ax.set_title('{}) '.format(chr(number+97)) +
                # '$C_{i,j}$ '+ rr.tag, **titleopts)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    tmp = 20 * np.log10(np.abs(rr.gammar))
                try:
                    tmp = np.array(
                        [
                            np.histogram(
                                y,
                                range=(
                                    np.floor(np.max([np.min(tmp), -30])),
                                    np.ceil(np.max(tmp)),
                                ),
                                bins=30,
                            )[0]
                            for y in tmp
                        ]
                    ).T
                    ## order after significance
                    # iorder = np.argsort(-np.mean(np.abs(rr.gammar), 1))  #order after significance
                    # axes[3].imshow(tmp[iorder,:], cmap = 'Greys', origin='lower')

                    axes[3].imshow(tmp, cmap="Greys", origin="lower")  # normal order
                    settitle(axes[3], 7, r"hist |$\gamma_k$|")
                    axes[3].set_xlabel("k")
                    axes[3].set_ylabel(r"$\left\Vert \gamma \right\Vert$ [dB]")
                    axes[3].grid()
                except:
                    pass

                # for ii in [0,1,2]:
                # axes[ii].set_yticks([])

                if not b_singlefig:
                    if b_saveplots:
                        plt.savefig(rr.title() + "_rec_stats.pdf")
                    if b_showplots:
                        plt.draw()
                    else:
                        plt.close()

            if b_singlefig == 1:
                fig.text( 0.05, 0.90, titlestring, transform=fig.transFigure, wrap=True,
                    usetex=False,)
            elif b_singlefig == 2:
                fig.text( 0.05, 0.94, 
                    # rr.title(), 
                    find_decomposition_clear(rr.decomposition),
                    transform=fig.transFigure, wrap=True, usetex=False,)

            if b_saveplots:
                if b_singlefig:
                    plt.savefig(rr.title() + ".pdf")
                    print(" > pdf", rr.title())
                else:
                    plt.savefig(rr.title() + "_basis_stats.pdf")

            if b_showplots:
                plt.draw()
            else:
                plt.close()

        rr.fit_timestamp = fstamp
        rr.__dict__.update({"id": rr.title(),})
        print("\t > coh: {:.2f}, nmse: {:.2f} dB, avg_nz: {:.2f}".format(
            rr.spatial_coherence , rr.nmse, rr.avg_nonzeros))


def plot_xy(figure, dframe,
    xkeys, ykey,
    select,
    xlims=None, xlabels=None, xlog=False, 
    ylims=None, ylabels=None, ylog=False,
    ):

    import matplotlib.pyplot as plt

    gridopts = { 'left': 0.12, 'right': 0.95, 'hspace': 0.60, }
    nsubs = len(select[1])
    if type(figure) is int:
        if nsubs == 2:
            fsize=(7, 7.5) # 2
            gridopts.update({
                'bottom': 0.09, 
                # 'top':    .82, #space for legend, else 95
                'top':    .80, #space for legend, else 95
                })
        elif nsubs == 3:
            fsize=(7, 10.5) # 3
            gridopts.update({
                'bottom': 0.08, 
                'top':    .86, #space for legend, else 95
                })
        else:
            fsize=(7, 13.0) # >3
        fig = plt.figure(figure, figsize=fsize) # 3
    elif type(figure) is tuple: 
        fig = plt.figure(1, figsize=figure)
    else: 
        fig = figure
    fig.clear()

    symbols = ["x", "o", "+", "*", "s", ">", "o", "v"]
    linestyles = ["-", "-.", ":", "--", "-.", "-.", "-.",":", "-", "-.", ":", "-.","-"]

    nx = len(xkeys)
    ny = len(select[1])
    decs = np.unique(dframe.decomposition, return_inverse=True)
    print(
            Decomposition['cpwe'].value,
            Decomposition['lpwe'].value,
            Decomposition['gpwe'].value,
            )

    gs0 = gs.GridSpec( ny, nx, figure = fig, **gridopts)
    axes = list()
    for yy, sk in enumerate(select[1]):
        for xx, xkey in enumerate(xkeys):
            ff = yy * nx + xx
            fidx = dframe[select[0]] == sk
            axes.append(fig.add_subplot(gs0[ff]))
            ax = axes[-1]

            for dd, dec in enumerate(decs[0]):

                #for grouped nof_waves
                # nofw = dec.split(' ')[1] # nwaves
                # dec = dec.split(' ')[0]  # nwaves

                idx = np.array((dd == decs[1]) & fidx)
                xd = dframe[idx].sort_values(xkey)[xkey]
                yd = dframe[idx].sort_values(xkey)[(ykey, "mean")]
                ys = dframe[idx].sort_values(xkey)[(ykey, "std")]
                ax.fill_between(
                    xd, yd - ys, yd + ys,
                    color="C{:d}".format(Decomposition[dec].value),
                    alpha=0.4, linewidth=0, zorder=2.1
                    )
                gg = ax.plot(
                    xd,
                    yd,
                    linewidth=2,
                    color="C{:d}".format(Decomposition[dec].value),
                    linestyle=linestyles[Decomposition[dec].value], 
                    # label=find_decomposition_clear(dec)+nofw, # nwaves
                    label=find_decomposition_clear(dec),
                    alpha=0.9,
                    # markersize=15,
                    # marker=symbols[dd],
                    # fillstyle='none',
                )
            ax.tick_params(which="both", direction="in")
            # ax.tick_params(which="minor", length=0)

            if xlims:   ax.set_xlim(xlims)
            if ylims:   ax.set_ylim(ylims)
            if xlog:    ax.set_xscale("log")
            if ylog:    ax.set_yscale("log")
            if ylabels: ax.set_ylabel(ylabels     )# + ", $\mu \pm \sigma$",usetex=False)
            else:       ax.set_ylabel(ykey.upper())# + ", $\mu \pm \sigma$",usetex=False)
            if 'gamma' in ylabels:
                if ((0>ylims[0])&(1.<ylims[1])):
                    ax.set_yticks([cc for cc in [.0, .5, .8, 1.]])
            if 'NMSE' in ylabels:
                ax.set_yticks([-20, -15,-10,-5,0])

            xt = dframe[xkey].sort_values().unique()
            if len(xt)>10: # find match for last idx only
                if ('spatial_sampling' in xkey):
                    if xlog:
                        xt = [5,10,20,40,80,160,320,640,1280]
                        # xt.append(4761)
                    else:
                        xt = [5,160,320,640,1000,1280]
                else:
                    xt = dframe[idx][xkey].sort_values().unique()
            xtls = np.round(xt).astype(int)
            if xkey == 'f':
                xt = [500, 600, 700, 800, 900, 1000, 1250, 1600, 2000]
                xtls = ['500',"",'700','','','1000','1250','1600','2000']
            ax.set_xticks(xt) 
            ax.set_xticklabels(xtls,)
                # horizontalalignment = 'right', rotation = 30) 

            if xlabels: ax.set_xlabel(xlabels[xx],usetex=False)
            else:       ax.set_xlabel(xkey.upper())

    return axes


def plot_mcmeasures(dframe):
    import matplotlib.pyplot as plt
    fig = plt.figure(4, figsize=(9, 9))
    fig.clear()

    gs0 = gs.GridSpec(2, 2, figure=fig, top=0.94, bottom=0.20)
    axes = [
        fig.add_subplot(gs0[0]),
        fig.add_subplot(gs0[1]),
        fig.add_subplot(gs0[2]),
        fig.add_subplot(gs0[3]),
    ]
    symbols = ["x", "d", "*", "s", ">", "o", "v"]
    freqs = [600, 800]
    decs = np.unique(dframe.decomposition, return_inverse=True)
    c = np.unique(dframe.spatial_sampling, return_inverse=True)[1]
    vmin = np.min(c)
    vmax = np.max(c)
    for ff, freq in enumerate(freqs):
        fidx = dframe.f == freq

        fidx = fidx & (dframe.spatial_sampling == 0)
        ax = axes[2 * ff]
        gridlist = []
        for dd, dec in enumerate(decs[0]):
            idx = (dd == decs[1]) & fidx
            ydata = dframe.nmse[idx]
            ax.scatter(
                dframe.spatial_coherence[idx],
                ydata,
                marker=symbols[dd],
                label=dec.upper(),
                alpha=0.4,
                cmap="Accent",
            )
        ax.legend(ncol=1)
        settitle(ax, 2 * ff, "Projection @ {:.0f} Hz".format(freq))
        ax.set_xlim([0.62, 1.02])
        ax.set_ylim([-21, 6])
        ax.set_ylabel(r"NMSE $<\epsilon_n^2>$ [dB]")
        if ff == len(freqs) - 1:
            plt.xlabel(r"$\gamma_\mathbf{\hat{p}p}$")
