''' generating figures for paper
'''

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
mpl.rcParams['font.size'] = 16
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.default'] = 'it'
mpl.rcParams['axes.facecolor']= 'white'
mpl.rcParams['axes.edgecolor']= 'white'
mpl.rcParams['axes.linewidth']= 0
mpl.rcParams['axes.grid']= True
mpl.rcParams['grid.color']= 'E5E5E5'
mpl.rcParams['grid.linestyle']= '-'
mpl.rcParams['savefig.dpi']= 300
mpl.rcParams['savefig.pad_inches']= 0
mpl.rcParams['savefig.bbox']= 'tight'
mpl.rcParams['xtick.labelsize'] = 'small'
mpl.rcParams['ytick.labelsize'] = 'small'


import numpy as np
from copy import deepcopy
import warnings
import os
storepath = os.path.abspath("./figures/decompositions/")
recpath = os.path.abspath("./figures/reconstructions/")
[os.makedirs(p, exist_ok=True) for p in [storepath, recpath]];

from sfr.sf import \
        Soundfield, \
        default_soundfield_config
from sfr.sfrec import \
        SoundfieldReconstruction,\
        default_reconstruction_config,\
        Decomposition,\
        find_decomposition_clear
from sfr.plot_tools import plot_gram_matrix, plot_dict2, add_cbar, annotatep
from sfr.tools import complex_interlace, norm_to_unit_std, norm_to_unit_abs


def plot_data_patches(soundfield, data = None):
    nof_samples = (10,10)
    grid        = (1,1)
    order       = ['real']
    dict_fsize  = (6.0,6.0)

    if not np.any(data):
        data = soundfield.fpatches[:,:,0] #multifreq
    data = complex_interlace(data, axis=0)
    data /= np.linalg.norm (data, ord=2, axis=1)[:, np.newaxis]
    nof_samples_total = np.prod(nof_samples)
    print("plot training data")
    rng = np.random.default_rng(1238908734)
    data = rng.choice(data,np.prod(nof_samples))

    plot_dict2(
        data.T,
        nof_samples,
        soundfield.psize,
        figure = dict_fsize,
        resolution=soundfield.dx / soundfield.wavelength,
        title="training data",
        # title="",
        savepath=os.path.join(storepath, "data_samples.pdf"),
        grid = grid,
        order = order,
    )
    plt.close()


def gen_dictionaries(
        declist = [
            # Decomposition.oldl,
            Decomposition.lpwe,
            Decomposition.gpwe,
            Decomposition.pca,
            # Decomposition.sinc,
            # Decomposition.ksvd,
            # Decomposition.rand
            ]):
    sfr_list = []

    for dec in declist:
        recopts, transopts = default_reconstruction_config(dec.name)

        # load or generate learning data, load objects

        trainopts = default_soundfield_config(measurement="011", frequency=600)
        learn     = Soundfield(**trainopts)
        if 'dl' in dec.name:
            # print("MULTIFREQ DICT")
            training_frequency = trainopts['frequency']
            recopts.update({'training_frequency' : training_frequency, 'random_state': 871623})
            data_frequencies = np.arange(590,610+1e-4,1.)
            trainopts.update({ 'frequency': data_frequencies, })
            learn = Soundfield(**trainopts)
            training_data = learn.multifreq_patches(target_freq = training_frequency)[0]
            if dec.name == 'oldl': # plot measured training data
                plot_data_patches(learn, training_data)
                if False: # debug
                    plot_data_patches(learn)

        sfrec_obj = SoundfieldReconstruction(
            training_field = learn, 
            transform_opts = transopts, 
            **recopts)
        sfrec_obj.fit()

        sfr_list.append(deepcopy(sfrec_obj))
    return sfr_list


def plot_dictionaries(sfrec_obj):
    # plot & save decomposition

    if 'pwe' in sfrec_obj.decomposition:
        grid = (1,2)
        order = ["real","imag"]
        nof_samples = None
    else:
        grid = (1,1)
        order = ['abs']

    order = ['real'] ## mod defence

    if 'pca' in sfrec_obj.decomposition:
        dict_fsize = (6,1.8)
        nof_samples = (2,10)
    else:
        dict_fsize = (6,4.6)
        nof_samples = (10,15)

    dictionary = None
    sfrec_obj._plot_dictionary(show=0,
            figure = dict_fsize,
            nof_samples = nof_samples,
            grid = grid,
            order = order,
            dictionary = dictionary,
            # title = " ",
            )
    plt.savefig( os.path.join(storepath, "decomposition_" + sfrec_obj.decomposition + "_dict.pdf"),)
    print(" > decomposition_" + sfrec_obj.decomposition)
    plt.close()

    # plot & save gram matrix
    fig = plt.figure(1, figsize=(4.9,4.1), clear=True)
    gs = fig.add_gridspec(1, 1, figure=fig, 
            left=0.10, bottom=0.04, right=0.85, top=0.79)
    ax = gs.subplots()
    if hasattr(sfrec_obj,"_A_learn_original"):
        overlay = sfrec_obj._shrink_dictionary(return_idx = True)
        nof_plot_atoms = int(np.ceil(np.sum(overlay)/15)+1)*15
        nof_plot_atoms = len(overlay)
        overlay = overlay[:nof_plot_atoms]
        D = sfrec_obj._A_learn_original[:,:nof_plot_atoms]
    else:
        D = sfrec_obj.A
        overlay = None

    _ = plot_gram_matrix(D, ax, 
            label = find_decomposition_clear(sfrec_obj.decomposition),
            overlay = overlay)
    plt.savefig( os.path.join(storepath, "decomposition_" + sfrec_obj.decomposition +
        "_gram.pdf"))
    print(" > decomposition_" + sfrec_obj.decomposition + "_gram")
    plt.close()


def plot_vars(sfr, color='k',overlay = True):
    if np.any(overlay):
        overlay = sfr._shrink_dictionary(return_idx = True)
        nof_plot_atoms = int(np.ceil(np.sum(overlay)/15)+1)*15
        nof_plot_atoms = len(overlay)
        overlay = np.array(overlay[:nof_plot_atoms])
        D = sfr._A_learn_original
    else:
        nof_plot_atoms = sfr.nof_components
        D = sfr.A

    D = D[:,:nof_plot_atoms]
    xb = np.arange(1,nof_plot_atoms+1)
    yb = sfr._explained_vars[:nof_plot_atoms]*100
    linestyles = ["-", "-.", ":", "--", "-", "--", ":", "-."]
    dec = Decomposition[sfr.decomposition]
    opts = {
                'alpha':0.9,
                'color':color,
                # 'color':"C{:d}".format(dec.value),
                'linewidth':2,
                'linestyle':linestyles[dec.value], 
                'label': find_decomposition_clear(dec),
            }
    if np.any(overlay):
        # plt.plot(xb[overlay], yb[overlay], **opts)
        plt.plot(xb, yb, **opts)
        plt.bar(xb[(overlay != True)], 100,color='r',alpha=.2)
        print(yb[-10:])
    else:
        plt.plot(xb, yb, **opts)
    plt.xlabel("$i$")
    plt.ylabel(r"accounted variance [%]\n"+r"$100 \% \, \left\Vert \mathbf{a}_i \otimes \mathbf{x}_i \right\Vert_F^2 \,/\,  \left\Vert \mathbf{P}_i \right\Vert_F^2$")
    plt.xticks([1,20,40,60,80,100,120,140,150])
    plt.xlim([1,150])
    plt.ylim([.5, 30.])
    plt.gca().set_yscale('log')


def plot_decomposition_hist(sfr_list):
    """ plot decomposition stats using the provided ...
        parameters:
               sfr_list    []      list of SoundfieldReconstruciton objects
    """

    nbins = 31
    lmin = -41
    lmax = 21

    l1 = np.log(10) / 10
    x = np.linspace(0 + lmin, 0 + lmax, 500)
    distribution = l1 * np.exp(l1 * (x - 0) - np.exp(l1 * (x - 0)))

    fig = plt.figure(2, figsize=(6.5, 3.1), clear=True)
    gs = fig.add_gridspec(1, 1, left=0.15, right=0.95, bottom=0.20, top=0.95)
    ax = gs.subplots()

   # consider statsmodels.api.qqplot_2samples(x,y)

    for sfrec_obj in sfr_list:
        decomposition = sfrec_obj.A # reconstruction 
        if 'pwe' not in sfrec_obj.decomposition:
            # decomposition = norm_to_unit_std(decomposition, axis=0)
            decomposition = decomposition.ravel()
        else:
            decomposition = decomposition.ravel()
            # decomposition = norm_to_unit_abs(decomposition, axis=0)

        ax.hist(
            20 * np.log10(np.abs(decomposition)),
            color="C{:d}".format(Decomposition[sfrec_obj.decomposition].value),
            bins=nbins,
            range=[lmin, lmax],
            density=True,
            alpha=0.9,
            histtype='step',
            linewidth = 3,
            label = find_decomposition_clear(sfrec_obj.decomposition),
            # label=sfrec_obj.tag.upper(),
        )
        print(sfrec_obj.decomposition, find_decomposition_clear(sfrec_obj.decomposition))
    ax.plot(x, distribution, "k", alpha=0.6, label="Random wave theory", linewidth=3)

    ax.legend(loc="upper left")
    ax.set_ylabel("histogram / density")
    ax.set_xlabel("sound pressure level [dB rel mean]")
    ax.set_ylim([0, 0.11])
    plt.savefig( os.path.join(storepath, "dictionary_hist.pdf"), dpi=300)
    print(" > dictionary_hist")
    plt.close()


def reconstruction_test(test_densities = [], sfrlist = None,
        valargs = dict(measurement = "019", frequency = 1000),mode='default',
        titles=None, b_plot_only=False, plot_intensity_idx=[],
        ):

    nl = len(sfrlist)
    nv, nh = len(test_densities)+len(plot_intensity_idx), 2+nl

    if mode == 'split':
        figs = [plt.figure(11+ii, figsize=(4.1,nv*3.3), clear=True) for ii in range(nh)]
        gs = [fig.add_gridspec(nv, 1, left=0.05, right=0.93, top=.85, bottom = .1,
                hspace = .8) for fig in figs]
        axes = np.concatenate([gsi.subplots() for gsi in gs])
    else:
        fig = plt.figure(11, figsize=(nh*2.8+1.2,nv*3.3), clear=True)
        gs = fig.add_gridspec(nv, nh, figure=fig, 
                left=0.05, right=0.93, hspace = .3, wspace = .6)
        axes = gs.subplots().flatten()

    valopts = default_soundfield_config(**valargs)
    val = Soundfield(**valopts)

    improp = dict(
        cmap   = 'viridis', 
        origin = 'lower', 
        extent =  val.extent ,
        )
    fontsize = 13

    # efunc = lambda x : np.abs(np.real(x))
    # clabel = '$\Re\{\mathbf{p}\}$ [dB rel $<{\mathbf{p}_{true}^2}>$]'
    efunc = lambda x : np.abs(x)
    if '019' in val.measurement:
        aopt = dict(xticks = [1.1,1.5,2.0,2.5], yticks = [3.0,3.5,4.0,4.5])
        xlabel = r'$x$ [m]'
        ylabel = r'$y$ [m]'
        clabel = r'[dB rel $<{\mathbf{p}_{true}^2}>$]'
        norm = 10 * np.log10(np.mean(efunc(val.p)**2))
        improp.update(dict(vmin=-20, vmax=10))
    else:
        clabel = '[dB SPL]'
        aopt = dict(xticks=[-2,0,3],yticks=[-1,0,2,4])
        xlabel = r'$x$ [$\lambda$]'
        ylabel = r'$y$ [$\lambda$]'
        norm = 20 * np.log10(2e-5)
        improp.update(dict(
            # vmin = np.min(20 * np.log10(efunc(val.p)))-norm,
            # vmax = np.max(20 * np.log10(efunc(val.p)))-norm,
            vmin = 50,
            vmax = 80,
            ))

    props = list()
    imdata = list()
    generate_titles = 0
    if not titles: 
        generate_titles = 1
        titles = list()
    vallabel = list()
    vectorfields = list()

    for dd, density in enumerate(test_densities):
        val.__dict__.update(dict(spatial_sampling = -density))
        if '019' in val.measurement: # demo case reproducability sampling
            val._seed = 387309083
            _ = val.sidx # generate common measurements

        if not b_plot_only:
            [sfr.clear_reconstruction() for sfr in sfrlist]
            [sfr.reconstruct(measured_field = val) for sfr in sfrlist]

        if '019' in val.measurement:
            vallabel.append(f"{val.frequency:.0f} Hz, "+r"$N_{\mathrm{obs}}$"+f"={val.N:.0f}\n")
        else:
            vallabel.append(r"$N_{\mathrm{obs}}$"+f"={val.N:.0f}\n")

        print("sound field f, num mics, density, surface ref norm", val.f, val.N, val.mic_density, norm)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            imdata.append( 20*np.log10(efunc(val.sp))  -norm)
            [imdata.append(20*np.log10(efunc(sfr.rf.p))-norm) for sfr in sfrlist]
            imdata.append( 20*np.log10(efunc(val.p))   -norm)
        if generate_titles:
            titles.append(r"$\mathbf{p}_{\mathrm{obs}}$ measurement")
            [titles.append(r"$\hat{\mathbf{p}}$ " + find_decomposition_clear(sfr.decomposition)) for sfr in sfrlist]
            titles.append(r'$\mathbf{p}$ true')
        if dd in plot_intensity_idx:
                imdata.append(20 * np.log10(efunc(val.sp))  - norm)
                [imdata.append(20 * np.log10(efunc(sfr.rf.IJ))) for sfr in sfrlist]
                imdata.append(20 * np.log10(efunc(val.IJ)))
                #indicate whcih plot function to use
                [vectorfields.append(ii) for ii in range(len(imdata)-nh+1,len(imdata))]
                if generate_titles:
                    titles.append(r"$\mathbf{p}_{\mathrm{obs}}$")
                    [titles.append(r"$\hat{\mathbf{I}}_{ux}$ " +
                        find_decomposition_clear(sfr.decomposition)) for sfr in sfrlist]
                    titles.append(r'$\mathbf{I}$ true')

        prop = np.array([(sfr.spatial_coherence, sfr.nmse, sfr.avg_nonzeros, sfr.rec_time) for sfr in sfrlist])
        props.append(prop)
        if hasattr(val,'sample_idx'): delattr(val,'sample_idx') # force re-sample sound field

    vallabel.reverse()
    for ii, data in enumerate(imdata):
        ax = axes[ii]
        if ii not in vectorfields:
            im = ax.imshow(data.squeeze(), **improp)
            ax.grid(False)
            cbar = add_cbar(im,ax)
            ax.set(**aopt)
            ax.set_title(titles[int(ii)], loc= "left", x=-.1, y= 1.3)
            subplothidx = ii%nh
            subplotvidx = int(np.floor(ii/nh))
            if (subplothidx == 0) :
                ax.set_ylabel(ylabel)
                annotatep(ax,vallabel.pop(), x = -.00, y = 1.0, align='left',color='k')
                annotatep(ax," \n({:.2f}".format(np.sqrt(test_densities[subplotvidx])) + 
                    r" mics per $\lambda$)", x = -.00, y = 1.0, align='left')
            elif (subplothidx<nh-1):
                annotatep(ax, 
                    r"$C\,=\,$"+"{:.2f}".format(props[subplotvidx][subplothidx-1,0]) +
                    "\n" + 
                    r"NMSE$\,=\,$"+"{:.2f} dB".format(props[subplotvidx][subplothidx-1,1]) ,
                    x = -.00, y = 1.01, align='left')
            if (subplothidx==nh-1):
                cbar.set_label(clabel)
            if (subplotvidx==nv-1):
                ax.set_xlabel(xlabel)
        else:
            Q = ax.quiver(data[0],data[1])
            qk = ax.quiverkey(Q, 0.6, 1.02, 5e-6, r'5$\,\mu$Wm$^{-2}$', labelpos='E')
            ax.set(xticks=[],yticks=[],aspect='equal')


    decs = [sfr.decomposition for sfr in sfrlist]
    tag = val.measurement+"_{:.0f}_".format(val.frequency)+"_".join(decs)
    if mode == 'split':
        for ff,fig in enumerate(figs):
            path = os.path.join(recpath,"rec_"+tag+"_"+str(ff)+".pdf")
            fig.savefig(path)
            plt.close()
    else:
        print(os.path.join(recpath, "rec_"+tag+".pdf"))
        fig.savefig(os.path.join(recpath, "rec_"+tag+".pdf"))
        plt.close()

    return sfrlist

def dl_paper():
    decompositions = [
        Decomposition.gpwe,
        Decomposition.lpwe,
        Decomposition.oldl,
        Decomposition.sinc,
        # Decomposition.pca,
        ]
    sfr_list = gen_dictionaries(declist = decompositions)
    freqs = [1000]

    [plot_dictionaries(sfr) for sfr in sfr_list if not "pwe" in sfr.decomposition]

    plt.figure(32,figsize=(6,4))
    plt.clf()
    sfr_list[2]._shrink_dictionary()
    sfr_list[2]._explained_vars = sfr_list[2].calc_explained_variances()
    sfr_list[4]._explained_vars = sfr_list[4].calc_explained_variances()
    plot_vars(sfr_list[2], color='k',overlay = True)
    plot_vars(sfr_list[4], color='grey',overlay = False)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(storepath, "decomposition_training_vars.pdf"),)
    plt.close()

    # DL paper figure 6 and 7, 10
    plist = [sfr for sfr in sfr_list if sfr.decomposition in ['oldl','pca']]
    [reconstruction_test([4.0, 12.0], plist,dict(measurement='019', frequency=freq)) for freq in freqs]
    sfr_list.extend(gen_dictionaries([Decomposition.sydl]))
    plist = [sfr for sfr in sfr_list if sfr.decomposition in ['sinc','sydl']]
    [reconstruction_test([4.0, 12.0], plist,dict(measurement='019', frequency=freq),mode='split') for freq in freqs]
    plist = [sfr for sfr in sfr_list if sfr.decomposition in ['gpwe','lpwe']]
    [reconstruction_test([4.0, 12.0], plist,dict(measurement='019', frequency=freq)) for freq in freqs]

    plt.close()


def plot_test():
    '''quickest reconstructions for plotting test run'''
    decompositions = [
        Decomposition.gpwe,
        Decomposition.gpwe,
        Decomposition.gpwe,
        Decomposition.gpwe,
        ]
    sfr_list = gen_dictionaries(declist = decompositions)
    reconstruction_test([4.0, 12.0], sfr_list, {'frequency':1000,'measurement':'019'} )
    plt.close()


if __name__ == "__main__":
    plot_test()
    dl_paper()
