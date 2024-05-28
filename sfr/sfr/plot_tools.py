import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

titleopts = {'loc' : 'left'}

def plot_dict(D, n_atoms, patch_size, figno, savepath=None, resolution = None,
        title = None, cmap = 'BrBG'):
    fig = plt.figure(figno)
    plt.clf()
    extent = (-.5, patch_size[0]+.5, -.5, patch_size[1]+.5)

    gs0 = GridSpec(int(np.ceil(n_atoms/10)),10, figure=fig, wspace=.15, hspace =.01)
    for ii in range(n_atoms-1,0-1,-1):
        ax = fig.add_subplot(gs0[ii])
        if np.imag(D[0,0]) == 0:
                im = ax.imshow(np.abs(D[:,ii]).reshape(patch_size),
                        cmap=cmap, 
                        interpolation='none',
                        extent = extent)
        else:
            if (np.prod(patch_size) > np.sum(patch_size)):
                im = ax.imshow(np.abs(D[:,ii]).reshape(patch_size),
                        cmap=plt.cm.gray_r, 
                        interpolation='none',
                        extent = extent)
                im = ax.imshow(np.angle(D[:,ii]).reshape(patch_size),
                        cmap=cmap, alpha=0.3, 
                        interpolation='none',
                        extent = extent)
            else:
                ax.plot(np.real(D[:,ii]), 'k')
                ax.plot(np.imag(D[:,ii]), 'k:')
                ax.set_aspect(patch_size[0])
                ax.spines['left'].set_visible(False)
                ax.set_yticks([])
        if not (ii==0):
            ax.axis('off')
            ax.patch.set_visible(False)
        else:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if resolution:
                lrange = np.array([0, .5, 1])
                prange = lrange * patch_size[0]
                ax.set_xticks(prange)
                ax.set_xticklabels([])
                # ax.set_xticklabels(["{:.1f}".format(y) for y in lrange])
                # ax.set_xlabel('$\lambda$',usetex=True)
                ax.set_yticks(prange)
                ax.set_yticklabels(["{:.1f}".format(y) for y in lrange], fontsize = 13)
                ax.set_ylabel(r'$\lambda$',fontsize = 13, usetex=False)
            else:
                prange = np.arange(0, np.floor(patch_size[0]/5)+1) *5
                ax.set_xticks(prange)
                ax.set_yticks(prange)
                ax.set_xticklabels(["{:.0f}".format(y) for y in prange])
                ax.set_yticklabels(["{:.0f}".format(y) for y in prange])

    if title is None:
        if resolution:
            title = (r"$\mathbf{D} = [\mathbf{d}_1, \mathbf{d}_2, \dots, \mathbf{d}_N]$"+
                    r", N = {:.0f}, side length {:.1f} $\lambda$ ({}x{})"
                    .format(D.shape[1], ((patch_size[0]-1)*resolution)[0], *patch_size))
        else:
            title = (r"D$ = [d_1, d_2, \dots, d_N]$"+" size {}x{}"
                    .format(D.shape[1], *patch_size))
    plt.text(.08,.90,title,transform=fig.transFigure, wrap=False)
    # plt.show()
    # if savepath != None:
        # plt.savefig(savepath)
    # else:
        # plt.savefig('decomposition_' + title.split('_')[0] + '.pdf')


def plot_dict2(D, K, patch_size, figure, savepath=None, resolution = None,
        title = None, cmap = 'twilight_shifted', 
        grid = (2,2), order = ['abs','real','phase','imag'], overlay = None):

    # dictionary dimensions
    n, K_total = D.shape

    # patch size
    n_r = patch_size[0]

    # patches per row / column
    if type(K) is tuple:
        K01 = np.array(K)
        K = np.prod(K)
    elif K < 200:
        K01 = np.array([int(np.ceil(K/10)), 10])
    else: 
        K01 = np.array([int(np.ceil(K/25)), 25])
    b_ylabel = int(K01[0] < K01[1])

    # we need n_r*K_r+K_r+1 pixels in each direction
    dim = n_r * K01 + K01 + 1
    V = np.zeros((dim[0], dim[1])) * np.max(D)
    V *= np.nan

    # compute the patches
    if K < K_total:
        selection_idx = np.random.choice(K_total,K)
    else:
        selection_idx = range(np.min([K_total,K]))
    patches = [np.reshape(D[:, i], (n_r, n_r)) for i in selection_idx]

    # place patches
    for i in range(K01[0]):
        for j in range(K01[1]):
            try:
                V[ i * n_r + 1 + i:(i + 1) * n_r + 1 + i ,
                   j * n_r + 1 + j:(j + 1) * n_r + 1 + j ] = \
                    patches[i * K01[1] + j]
            except:
                pass

    if type(figure) is int:
        fig = plt.figure(figure, figsize=(6,3))
    elif type(figure) is tuple:
        fig = plt.figure(1, figsize=figure)
    else: # figure handle
        fig = figure
    plt.clf()

    if b_ylabel:
        gs0 = GridSpec(*grid, figure=fig, 
                hspace = .16,
                top = .90, bottom = .02,
                left = .05, right = .98)
    else:
        gs0 = GridSpec(*grid, figure=fig, 
                wspace = .30,
                top = .90, right = .95,
                )

    axes = []
    for gs, quantity in zip(gs0, order):
        axes.append(fig.add_subplot(gs))
        if quantity.lower() == 'abs':
            im = axes[-1].imshow(np.abs(V), 
                    cmap = 'binary', 
                    vmin = 0,
                    )
            label_text = r"$|\cdot|$" # Mag
            # cbar = add_cbar(im,axes[-1])
        elif quantity.lower() == 'real':
            im = axes[-1].imshow(np.real(V), 
                    cmap='RdBu', 
                    )
            label_text = r"$\Re$"
            # cbar = add_cbar(im,axes[-1])
        elif quantity.lower() == 'phase':
            im = axes[-1].imshow(np.angle(V)/np.pi, 
                    cmap='twilight_shifted', alpha=1.0, 
                    vmin=-1, vmax=1, 
                    )
            label_text = r"$\angle$" # Phase
            # cbar = add_cbar(im,axes[-1])
            # cbar.set_ticks([-1, -.5, 0, .5, 1])
            # cbar.ax.set_yticklabels([r'$-\pi$',r'$-\pi/2$',
               # r'0',r'$+\pi/2$',r'$+\pi$'], usetex=False)
        elif quantity.lower() == 'imag':
            im = axes[-1].imshow(np.imag(V),
                    cmap='BrBG', 
                    alpha=1.0, 
                    )
            label_text = r"$\Im$"
            # cbar = add_cbar(im,axes[-1])

        if np.any(overlay):
            ol_mask = np.zeros((dim[0], dim[1])) * np.max(D)
            for i in range(K01[0]):
                for j in range(K01[1]):
                    try:
                        ol_mask [ i * n_r + 1 + i:(i + 1) * n_r + 1 + i ,
                           j * n_r + 1 + j:(j + 1) * n_r + 1 + j ] = \
                            1-overlay[i * K01[1] + j]
                    except:
                        pass
            axes[-1].imshow(ol_mask*.5, 
                    alpha = ol_mask*.2, 
                    cmap = 'Reds',
                    vmin = 0, vmax = 1)

        # if b_ylabel:
            # plt.text(-.08,.85,label_text,transform=axes[-1].transAxes,
                    # fontsize=22, wrap=False)
            # # axes[-1].set_ylabel(label_text, rotation=0, fontsize=16)
        # else:
            # axes[-1].set_title(label_text, fontsize=16)

        # ticks
        xticks = [ii*(n_r+1) for ii in range(K01[1]+1)]
        yticks = [ii*(n_r+1) for ii in range(K01[0]+1)]

        # axes[-1].set_xticks([])
        # axes[-1].set_yticks([])

        axes[-1].tick_params(length=5, color="#555555")
        axes[-1].tick_params(axis="x", 
                bottom=False, labelbottom=False,
                top   =True , labeltop=True)
        axes[-1].set_xticks(xticks[:2])
        axes[-1].set_yticks(yticks[:2])
        axes[-1].set_xticklabels(["0",r"$\lambda$"], )
        axes[-1].set_yticklabels(["0",r"$\lambda$"], )
        axes[-1].grid(0)

        # axes[-1].tick_params(length=12, color="#555555")
        # axes[-1].set_xticks(xticks)
        # axes[-1].set_yticks(yticks)
        # if (len(axes)/grid[1]+1) > grid[0]:
            # axes[-1].set_xticklabels([r"  $\lambda$"] * K01[1], 
                    # horizontalalignment = 'left',
                    # verticalalignment = 'center',
                    # position = (0,+.03),
                    # usetex=False,
                    # )
        # else:
            # axes[-1].tick_params(axis='x', length=3, color="#cccccc")
            # axes[-1].set_xticklabels([])

        # if (len(axes)-1)%grid[1] == 0:
            # axes[-1].set_yticklabels([r"  $\lambda$"] * K01[0], 
                    # horizontalalignment = 'left',
                    # verticalalignment = 'bottom', 
                    # position = (+.0,-0),
                    # rotation=90,
                    # usetex=False,
                    # )
        # else:
            # axes[-1].tick_params(axis='y', length=3, color="#cccccc")
            # axes[-1].set_yticklabels([])

    if title is None:
        if resolution:
            title = (r"$\mathbf{D} = [\mathbf{d}_1, \mathbf{d}_2, \dots, \mathbf{d}_N]$"+
                    r", N = {:.0f}, side length {:.1f} $\lambda$ ({}x{})"
                    .format(D.shape[1], ((patch_size[0]-1)*resolution)[0], *patch_size))
        else:
            title = (r"D$ = [d_1, d_2, \dots, d_N]$"+" size {}x{}x{}"
                    .format(D.shape[1], *patch_size))

    # if title & (K < K_total): title = " ".join([title,"({:.0f}/{:.0f})".format(min(K,K_total), K_total)])
    plt.text(.5,.97,title,ha='center',va='top', transform=fig.transFigure, wrap=False)

    if savepath != None:
        if 'pdf' in savepath:
            plt.savefig(savepath)
    else:
        plt.savefig('decomposition_' + title.split(' ')[0].split('_')[0] + '.pdf')


def plot_gram_matrix(A, ax, label=None, overlay = None):
    ''' plot coherence of columns in matrix A

    parameters:
        A   matrix
        ax  matplotlib axes
        label annotation (optional)

    A
    plots GA = np.abs(np.dot(A.conj().T,A))
    '''
    # numpy does not detect hermitian transposes when
    # computing dot product -> symmetry only used in real transpose case.

    Atrans = A.conj().T 
    GA = np.abs(Atrans.dot(A))
    mutual_coherence = np.max(GA-np.diag(np.diag(GA)))

    im = ax.imshow(GA, cmap = 'Greys',
            extent = [.5, GA.shape[0]+.5]*2,
            vmin = .0, vmax = 1)

    if np.any(overlay):
        mask = ~np.array(overlay)
        ol_mask = np.zeros(GA.shape)
        ol_mask[mask,:] = 1
        ol_mask[:,mask] = 1
        ax.imshow(ol_mask*.5,
                alpha = ol_mask*.2, 
                  # cmap = 'Reds',
                  cmap = 'Greys',
                extent = [.5, GA.shape[0]+.5]*2,
                vmin = 0, vmax = 1)

    if A.shape[1]>100:
        dt = 25
    else:
        dt = 10
    ticks = [1] + [r for r in range(dt,A.shape[1],dt)] + [A.shape[1]]
    ax.grid()
    ax.tick_params(axis="x", 
            bottom=False, labelbottom=False,
            top   =True , labeltop=True)
    ax.set_xticks(ticks)
    ax.set_ylabel(r"$i$",fontsize=16, rotation=0)
    ax.set_yticks(ticks)
    ticks.reverse()
    ax.set_yticklabels(ticks)
    ax.set_title(r'$j$',fontsize=16)
    cbar = add_cbar(im,ax)
    ax.text(
        -.2,
        1.20,
        r'Gram matrix $\mathbf{C}_{i,j}$, '+ label, 
        transform=ax.transAxes,
        wrap=False,
        horizontalalignment="left",
        )
    ax.text(
        1.34,
        1.20, #-.02, #1.05
        r"$\mu_{\mathbf{C}} =$" + ' {:.3f}'.format(mutual_coherence),
        # "$\mu = \max_{i != j} \mathbf{C}_{ij} =$"
        # "$\mu = max \{ \mathbf{C}_{i != j} \} =$"
        transform=ax.transAxes,
        wrap=False,
        color="#888888",
        horizontalalignment="right",
        fontsize=18,
    )
    return ax


def add_cbar(im, ax, label=None):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    # if (ax.figure.bbox_inches._points[1,0]>8):
    axsize = ax.bbox._bbox._points_orig
    if ((axsize[1,0]-axsize[0,0]) > .5):
    # if (ax.figure.bbox_inches._points[1,0]>8):
        cax = divider.append_axes('right', size='4%', pad=0.04)
    else:
        cax = divider.append_axes('right', size='4%', pad=0.04)
    cb =  plt.colorbar(mappable = im, cax=cax)
    if label: cb.set_label(label)
    return cb


def annotatep(handle, text, x=1.15, y=1.30, usetex = None, align='right',color="#666666"):
    transform = handle.transAxes
    handle.text(
        x,
        y,
        text,
        transform=transform,
        wrap=False,
        color=color,
        horizontalalignment=align,
        verticalalignment = "bottom",
        usetex = usetex,
    )


def plot_subplots(axes, ppa, tag, dims = None, vrange = [75,125]):

    for ii, pa in enumerate(ppa):
        improp = {
                'origin': 'lower',
                }
        if np.ndim(vrange) == 1:
            improp.update({'vmin'  : vrange[0], 'vmax'  : vrange[1],})
        elif np.ndim(vrange) == 2:
            improp.update({'vmin'  : vrange[ii,0], 'vmax'  : vrange[ii,1],})
        if dims.any():
            aperture = np.array(dims).flatten()[:4]
            dx_2     = np.diff(aperture[:2])/(pa.shape[0]-1)/2
            dy_2     = np.diff(aperture[2:])/(pa.shape[1]-1)/2
            improp['extent'] = tuple((aperture + np.array([-dx_2, dx_2, -dy_2, dy_2]).T)[0])
        if ('angle' in tag[ii]): # angle
            pdata = np.angle(pa)/np.pi*180
            improp['cmap'] = 'BrBG'
            improp['vmin'] = -180
            improp['vmax'] = 180
            if (tag[ii] in ['-','diff','error']):
                improp['cmap'] = 'RdBu'
                for idx in [tuple(ii) for ii in np.argwhere(pdata>180)]:
                    pdata[idx] -=360
                for idx in [tuple(ii) for ii in np.argwhere(pdata<-180)]:
                    pdata[idx] +=360
        elif 'code' in tag[ii]:
            pdata = 20*np.log10(np.abs(pa))
            improp['vmin'] = -10
            improp['vmax'] = None
        elif len(tag[ii]) == 0:
            continue
        else:
            with np.errstate(divide='ignore'):
                # pdata = 20*np.log10(np.abs(pa))
                pdata = pa
            # if 'diff' in tag[ii]:
            if ii == 3:
                improp['cmap'] = 'Greys'
                improp['vmin'] = -30
                improp['vmax'] =  0
            if ii > 3:
                improp['cmap'] = 'Greys'
                [improp.pop(elem) for elem in ['vmin','vmax']];

        with np.errstate(divide='ignore'):
            im = axes[ii].imshow(pdata, **improp)
        # axes[ii].set_title("{}) ".format(chr(97+ii)) + tag[ii], loc='left')
        axes[ii].grid()
        cbar = add_cbar(im,axes[ii])
        if (ii<3):
            # cbar.set_label('[dB rel $<{\mathbf{p}_{ref}^2}>$]')
            cbar.set_label('[dB rel 1 Pa]')
        elif (ii==3):
            cbar.set_label(r'[dB rel $\mathbf{p}_{ref}$]')
        axes[ii].set_xlabel('x [m]')
        axes[ii].set_ylabel('y [m]')
        # else:
            # axes[ii].set_xticks([])
            # axes[ii].set_yticks([])
    # foot()
