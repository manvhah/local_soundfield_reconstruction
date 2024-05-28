import numpy as np
import time
import warnings
from enum import Enum, auto
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

from copy import deepcopy

from sfr.const import c0
from sfr.sf import Soundfield, default_soundfield_config, reconstruct_from_patches_2d, scale_patches
from sfr.inverse_problem import solve_ip
from sfr.sphere_sampling import fibonacci_sphere
from sfr.tools import complex_interlace, complex_delace, augment_2d_data, calc_coherence
from sfr.plot_tools import plot_dict2
from sfr.dictionaries import sinc_kernel_dictionary, random_dictionary
from sfr.sfr_utils_hdf import *

class Decomposition(Enum):
    gpwe = 0
    lpwe = auto()
    pca  = auto()
    oldl = auto()
    sinc = auto()
    sydl = auto()

DECOMPOSITION      = ['gpwe', 'lpwe', 'pca', 'oldl', 'sinc', 'sydl']
TRAINING           = ['011', '019', '1p011', '1p019', 'mono', '/work1/manha/z_data/room011.h5']
EVALUATION         = ['011', '019', '1p011', '1p019', 'sim', '/work1/manha/z_data/019_lecture_room.h5']
REC_FREQ           = [500, 600, 700, 800, 900, 1000, 1250, 1600, 2000]
SPATIAL_SAMPLING   = [ -4., 40, 80, 160, 320,-12., 640,1280,0.0, ]

# float for spatial_sampling,
# int for N of mics, neg float for density per circle of wavelength diameter

ALLPARAMETERS      = [
        DECOMPOSITION,    TRAINING,
        EVALUATION,       REC_FREQ,
        SPATIAL_SAMPLING,
        ]

class Decomposition_clear(Enum):
    lpwe = "Local independent"# plane waves"
    gpwe = "Global"# "PWE"#"global plane waves"
    pca  = "PCA"
    oldl = "Dictionary Learning"#"Learned on measured data"#"DL" # local learned dictionary
    sinc = "Synthesized"
    sydl = "Learned on synthetic data"

def decompositions(): return Decomposition
def decompositions_clear(): return Decomposition_clear
def find_decomposition_clear(dec):
    if type(dec) == Decomposition:
        return Decomposition_clear[dec.name].value
    elif type(dec) == str:
        return Decomposition_clear[dec].value

def spatial_coherence(x,y):
    return calc_coherence(x, y)

def assess_sparsity(gamma, num_patches=None, tol=.001, nof_components = False,
        return_std=False):
    nnonzeros = np.mean(np.abs(gamma)>0)*gamma.shape[0]
    if return_std:
        return nnonzeros, np.std(np.abs(gamma)>0)*gamma.shape[0]
    else:
        return nnonzeros
    
    # # approximation_norm = np.linalg.norm(gamma,2,axis=0)
    # approximation_norm = np.linalg.norm(gamma)
    # if gamma.squeeze().ndim == 1:
        # return np.sum(np.abs(gamma) > tol*approximation_norm)
    # elif gamma.ndim == 2: 
        # nonzeros = (np.abs(gamma)> tol*approximation_norm)
        # # nonzeros = (np.abs(gamma)> tol*approximation_norm[np.newaxis,:])
        # return np.mean(np.sum(nonzeros,axis=0))

def mse(x,reference):
    # with warnings.catch_warnings():
        # warnings.simplefilter("ignore", RuntimeWarning)
    x = x.ravel()
    y = reference.ravel()
    # averaged magnitude and phase errors, standard
    return 20 * np.log10(np.linalg.norm(x-y))

def nmse(x,reference):
    # with warnings.catch_warnings():
        # warnings.simplefilter("ignore", RuntimeWarning)
    x = x.ravel()
    y = reference.ravel()
    # pointwise comparison of magnitude and phase errors 
    # return 10 * np.log10(np.mean((np.abs(x - y) / np.abs(y)) ** 2)) 
    # averaged magnitude and phase errors, standard
    return mse(x,y)-20 * np.log10(np.linalg.norm(y))

def checkdelattr(dictionary, attribute): 
    """remove attribute if exist, checkdelattr(obj,attr)"""
    if hasattr(dictionary, attribute):
        delattr(dictionary, attribute)

def checkmoveattr(dictionary, attribute, newattr): 
    """if exists, move attribute to , checkmoveattr(obj,attr,newattr)"""
    if hasattr(dictionary, attribute):
        dictionary[newattr] = dictionary[attribute]
        delattr(dictionary, attribute)

class SoundfieldReconstruction():
    '''
    todo:
    - move functions in
    - make functions lazy, only link to data at the highest layer
    '''
    def __init__(self,
            measured_field = None,
            training_field = None,
            **kwargs
            ):
        ''' TODO: just unpack all args and check if exist'''
        self.measured_field = measured_field
        self.training_field = training_field
        self.reconstructed_field  = None # no reconstruction available yet
        self.correlatecomplex = False
        self.__dict__.update(kwargs)

    def title(self):
        if self.mf.spatial_sampling == 0.0: # integer number of mics
            samplingstr =  "{:0>5.0f}".format(np.prod(self.rf.shp))
        elif self.mf.spatial_sampling%1: # integer number of mics
            samplingstr =  "{:_<5.3f}".format(
                self.mf.spatial_sampling).replace('.','p')
        elif (self.mf.spatial_sampling >= 0.0): # lossfactor 0..1
            samplingstr =  "{:0>5.0f}".format(self.mf.spatial_sampling)

        if hasattr(self, 'rec_timestamp'):
            rt = '{:0>5d}'.format(int(self.rec_timestamp%1e5))
        else:
            rt = ''

        return '_'.join([
            "{:_<5}".format(self.decomposition),
            # self.mf.measurement,
            '{:0>4.0f}'.format(self.mf.frequency),
            samplingstr,
            # self.training_field.measurement,
            str(int(self.fit_timestamp%1e8)),
            rt,
            ])

    def _gen_plane_wave_expansion(self, soundfield, 
            particle_velocity = False, 
            force_global = False):
        k_abs = 2*np.pi*soundfield.frequency/c0
        self._ksphere = fibonacci_sphere(samples = self.nof_components)
        self._ksphere[:,2]*=np.sign(self._ksphere[:,2]) # kz pos for planar field
        self.fit_timestamp = int(time.time())
        if not self.decomposition in ['lpwe'] or force_global:
            funcsize = soundfield.paddedshape
        else:
            funcsize = soundfield.psize

        rx = np.arange(funcsize[0])
        ry = np.arange(funcsize[1])
        rr = np.dstack(np.meshgrid(rx,ry,0))
        rr = rr.reshape((-1,3)) *soundfield.dx
        A     = np.exp(k_abs * 1j*rr @ self._ksphere.T)/np.sqrt(len(rr))

        if particle_velocity:
            A = A[...,np.newaxis]*self._ksphere[np.newaxis,...]/(1.2*c0)
        return A

    def _online_dl(self, training_data, dict_init = None):
        from sklearn.decomposition import MiniBatchDictionaryLearning
        if False:
            dict_init = sinc_kernel_dictionary(self.tf.psize,
                    self.tf.k*self.tf.dx, self.nof_components).T
        if hasattr(self,'random_state'):
            random_state = self.random_state
            print("DL random state",random_state)
        else:
            random_state = None

        if self.correlatecomplex: #covariates
            self._training_data = complex_interlace(training_data, axis=1)
            dict_init = complex_interlace(dict_init,1)
        else: #independent Re Im, more samples from complex training_data
            self._training_data = complex_interlace(training_data,0)
        num_training_samples = self._training_data.shape[0]
        print(num_training_samples, " training samples")

        onlinedl = MiniBatchDictionaryLearning(
                dict_init = dict_init,
                transform_algorithm = 'lasso_lars',
                n_components = self.nof_components,
                alpha        = self.alpha,
                max_iter     = self.n_iter,
                batch_size   = self.batch_size,
                # n_jobs     = 4,
                # shuffle    = True,
                # verbose    = True,
                random_state = random_state,
                fit_algorithm = self.fit_algorithm,
                )

        self.dlstats = dict({})
        if True: # automatic batch fitting, e.g.
            onlinedl.fit(self._training_data)
        else:
            self.dlstats.update({
                'learningval_nmse' : list(),
                'learningval_nnonzeros' : list(),
                })
            idx_eval = np.random.choice(num_training_samples,1000,replace=False)
            Yeval = self._training_data[idx_eval,:]
            Ylearn = np.delete(self._training_data,idx_eval,0)
            Ylearn = self._training_data

            for ii in range(self.n_iter):
                # fit
                onlinedl.partial_fit(Ylearn[np.random.choice(Ylearn.shape[0],self.batch_size,replace=False),:])
                # plot stats
                if (ii>1):
                    X_fit = onlinedl.transform(Yeval)
                    Y_fit = X_fit @ onlinedl.components_
                    self.dlstats['learningval_nmse'].append(nmse(Y_fit,Yeval))
                    self.dlstats['learningval_nnonzeros'].append(assess_sparsity(X_fit.T))
                    # if ii%100:
                        # print("{:3.1f}".format(ii/self.n_iter*100), 
                                # "%,\t nmse [dB]",
                                # self.dlstats['learningval_nmse'][-1], 
                                # ",\t nnonzeros",
                                # self.dlstats['learningval_nnonzeros'][-1])
            # try:
                # plt.figure(321)
                # plt.subplot(121)
                # plt.plot(self.dlstats['learningval_nmse'], label="{}_{}".format(self.n_iter,self.batch_size))
                # plt.subplot(122)
                # plt.plot(self.dlstats['learningval_nnonzeros'], label="{}_{}".format(self.n_iter,self.batch_size))
                # plt.savefig("dl_a{}_n{}_i{}_times_b{}.pdf".format(
                    # onlinedl.alpha,
                    # onlinedl.n_components,
                    # self.n_iter,
                    # self.batch_size
                    # ))
            # except:
                # pass

        # DL stats
        self._X_fit = onlinedl.transform(self._training_data)
        Y_fit = self._X_fit @ onlinedl.components_

        self.dlstats.update({ 'nnonzeros' : 
            assess_sparsity( self._X_fit.T),
            'nmse' : nmse(Y_fit,self._training_data), 
            'explained_variance' : (np.linalg.norm(Y_fit)/ np.linalg.norm(self._training_data)) ** 2,
            })
        print("\n > DL avg_nnzeros", self.dlstats['nnonzeros'],
              "\t, nmse"           , self.dlstats['nmse'],
              "\n > DL batch size" , self.batch_size,
              "\t, n_iterations"   , self.n_iter,
              "\t, sparse coding with ",self.fit_algorithm,
              "\n > alpha"         , self.alpha,
              "\t, N_of_atoms"     , self.nof_components,
              "\t, sparse coding with ",self.fit_algorithm,
              )

        dictionary = onlinedl.components_.T
        self._dlobj = onlinedl # for debugging
        if self.correlatecomplex: 
            dictionary = complex_delace(dictionary)
        else: dictionary = dictionary.astype(float)


        # analyse fit, sort
        amplitude_norms = np.linalg.norm(self._X_fit,2,axis=0)/np.linalg.norm(self._X_fit)
        sortidx = np.argsort(amplitude_norms)[::-1]
        self.A_learn = dictionary[:,sortidx]
        self._amplitude_norms = amplitude_norms[sortidx]
        self._X_fit = self._X_fit[:,sortidx]
        self._A_learn_original = self.A_learn
        self._explained_vars = self.calc_explained_variances()
        # self._plot_dictionary(2, title='learned dictionary')

        # shrink if not used in training data or if far away form mutual coherence
        self._shrink_dictionary() # coherence more reliable

        print("Dict shrinked to {} atoms".format(self.nof_components))
        # self._plot_dictionary(3, title='effective dictionary')


    def calc_explained_variances(self):
        # explained variance, amplitude norms
        try:
            Y = self._training_data
            X = self._X_fit
            if hasattr(self,'_A_learn_original'):
                A = self._A_learn_original
            else:
                A = self.A_learn
            threshold = .01*np.linalg.norm(X,axis=1)

            def calc_normvariance(x,d):
                select_samples = (x>threshold) #non-sparse
                diff = np.linalg.norm(np.outer(x[select_samples],d))
                return (diff/np.linalg.norm(Y[select_samples]))**2

            vars = [calc_normvariance(x,d) for x,d in zip(X.T,A.T)]
            return np.array(vars)
        except:
            warnings.WarningMessage('explained var calc fail: training coeff not available')
            pass


    def _shrink_dictionary(self, return_idx = False, 
            # mode = "coherence"):
            # mode =  "training_coefficients"):
            mode = "explained_variance"):
        D = self._A_learn_original

        if mode=='coherence': #set threshold and sort after coherence
            num_atoms = D.shape[1]

            # sort atoms again
            GmI = D.T.dot(D)-np.eye(num_atoms)
            sortidx = np.argsort([max(aa**2) for aa in GmI])
            D = np.flip(D[:,sortidx], axis=1)
            self._A_learn_original = D

            GmI = D.T.dot(D)-np.eye(num_atoms)
            mu = np.max(GmI)
            keepidx = np.array([max(aa**2)>(.3*mu**2) for aa in GmI])
        else: # use training coefficients to set threshold at 1%
            if mode == "explained_variance":
                criterion = self._explained_vars
                threshold = .001
            if mode == "training_coefficients":
                criterion=self._amplitude_norms
                threshold = .02
            # .4%
            keepidx = (criterion>threshold)
            # find significant jump
            # keepidx = (self._amplitude_norms>.0001)
            # nump = np.argmax(np.abs( np.diff(self._amplitude_norms[10:])/ self._amplitude_norms[10:-1]))
            # keepidx[jump+1:] = False

        self.A_learn = D[:, keepidx]
        self.nof_components = sum(keepidx)

        if return_idx:
            return keepidx


    def _scale_basis_functions(self):
        A_scaled = scale_patches(self.A.T, 
                self.tf.dx, self.tf.frequency,
                self.mf.dx, self.mf.frequency, 
                self.tf.patch_size_in_lambda,
                # mode='direct' # not for DL PAPER
                ).T
        A_scaled /= np.linalg.norm(A_scaled, axis=0)[np.newaxis,:]
        self.A_rec = A_scaled


    def fit(self, field = None):
        if field:
            self.training_field = field
        # TODO: update flag when settings change, like val.frequency
        tic = time.time()

        # prepend _ to avoid storage in hdf
        if 'pwe' in self.decomposition:   # global
            if self.tf is None:
                self.training_field = Soundfield('mono') #random config
            self._A_learn = self._gen_plane_wave_expansion(self.tf)
        else:
            if 'sydl' in self.decomposition:
                # for DL on synthetic data, copy 011 measurement layout
                self.tf.measurement = 'synthesized_diffuse'
                self.tf.dx = .05 
                self.tf.nof_apertures = 7
                self.tf.rdim = [[0, .81],[0, .81],[0,0]]
                self.tf.rdim = np.array(list(self.tf.extent)+[0,0]).reshape((-1,2))
                self.tf._reset()
            if hasattr(self,'training_frequency'): # this should be patch_extraction_frequency in Soundfield()
                # use training_field patches from all frequencies and scale to training frequency
                training_data, training_psize  = self.tf.multifreq_patches(target_freq = self.training_frequency)
                training_data  = augment_2d_data(training_data, training_psize)
                training_data_f = (np.min(self.tf.f),np.max(self.tf.f),np.diff(self.tf.f)[0])
                print('{:_<50}'.format(' > {} train at [{:.2f}:{:.2f}:{:.2f}] Hz on {} complex patches in'.format(
                    self.decomposition, *training_data_f, training_data.shape[0])), end='_')
            else:
                training_data = augment_2d_data(self.tf.patches,self.tf.psize)
                training_data_f = self.tf.frequency
                print('{:_<50}'.format(' > {} train at {:.2f} Hz on {} patches in'.format(
                    self.decomposition, training_data_f, training_data.shape[0])), end='_')

            if self.decomposition in ['oldl','sydl']:
                print("training data mean", np.mean(np.linalg.norm(training_data,axis=1)))
                self._online_dl(training_data)
            elif 'pca'   in self.decomposition:
                from sklearn.decomposition import PCA
                # noverlay = np.ceil(self.nof_components/10)*10
                # pca_overlay = PCA(noverlay).fit(complex_interlace(training_data))
                # self._A_learn_original = pca_overlay.components_.T #illustrating truncated components

                self._training_data = complex_interlace(training_data)

                self.nof_components = min([self.nof_components,np.prod(self.tf.psize)])
                pca = PCA(n_components=self.nof_components)
                pca.fit(self._training_data)
                self.A_learn          = pca.components_.T.astype(float)
                self.nof_components   = self.A_learn.shape[1]

                self._X_fit           = pca.transform(self._training_data)
                self._explained_vars2 = pca.explained_variance_ratio_
                self._explained_vars  = self.calc_explained_variances()

                print("\ncomponents: ",self.nof_components,", explained variance", sum(pca.explained_variance_ratio_))

            elif 'sinc'  in self.decomposition:
                self.A_learn = sinc_kernel_dictionary(self.tf.psize,
                        self.tf.k*self.tf.dx, self.nof_components, gen_factor=1)

            del training_data
            print('{: >5.2f}'.format(time.time()-tic))

        self.fit_timestamp = int(time.time())

        if self._figno != None:
            self._plot_dictionary(self._figno)

    def _plot_dictionary(self, figure = 1, show=False, 
            nof_samples = None, 
            title = None,
            order = None, #['abs','real','phase','imag'],
            grid = None,
            dictionary = None):
        if order is None:
            if np.all(np.imag(self.A) == 0): order = ['real']
            else: order = ['real','imag']
        if dictionary is None:
            dictionary = 'learn'
        if not grid: grid = (len(order),1)
        if not nof_samples: nof_samples = self.nof_components
        if not title: title = Decomposition_clear[self.decomposition].value

        D = self.A
        field = self.training_field
        overlay = None
        if dictionary == 'rec': 
            D = self.A_rec
            field = self.measured_field
        if hasattr(self,'_A_learn_original'):
            if self.decomposition in ['oldl','sydl']:
                overlay = self._shrink_dictionary(return_idx = True)
            elif self.decomposition == 'pca':
                overlay = np.hstack((np.ones(self.nof_components), 
                    np.zeros(20-self.nof_components)))
            nof_plot_atoms = int(np.ceil(np.sum(overlay)/12)+1)*12
            overlay = overlay[:nof_plot_atoms]
            D = self._A_learn_original[:,:nof_plot_atoms]

        plot_dict2(D,
                nof_samples, 
                field.psize, 
                figure,
                resolution = field.dx/c0*field.frequency,
                # title = title,
                title = " ".join([title, dictionary,]),
                grid = grid,
                order = order,
                overlay = overlay,
                )

        if show:
            plt.draw()
            plt.show()

    def _gen_A_rec(self):
        '''from the fitted A, generate a suitable reconstruction transfer matrix '''

        if  'pwe' in self.decomposition:
            self._A_rec = self._gen_plane_wave_expansion(self.measured_field)
            self._Au_rec = self._A_rec[np.newaxis,...]*self._ksphere.T[:,np.newaxis,:]/(1.2*c0)
        elif 'sinc' in self.decomposition: # on sparse field
            self.A_rec = sinc_kernel_dictionary(self.mf.psize,
                    self.mf.k*self.mf.dx, self.nof_components)
            self.fit_timestamp = int(time.time())
        else: # scale learned dictionary
            if (self.mf.frequency != self.training_field.frequency) or (self.mf.dx != self.training_field.dx):
                self._scale_basis_functions()
            else:
                self.A_rec, self.mf.psize = self.A, self.training_field.psize
        # self._plot_dictionary(5, dictionary='learn')
        # self._plot_dictionary(6, dictionary =  'rec')

    @property
    def A(self):
        if (not hasattr(self,'A_learn')) and (not hasattr(self,'_A_learn')):
            self.fit()
        if hasattr(self,'A_learn'):
            return self.A_learn
        else:
            return self._A_learn

    @property
    def Ar(self):
        if self.decomposition == 'swin':
            return np.zeros(1)
        if (not hasattr(self,'A_rec')) and (not hasattr(self,'_A_rec')):
            self._gen_A_rec()
        if hasattr(self,'A_rec'):
            return self.A_rec
        else:
            return self._A_rec

    @property
    def Aur(self):
        if hasattr(self,'_Au_rec'):
            return self._Au_rec
        else:
            return self.Ar

    @property
    def coeffs(self):
        if (not hasattr(self,'coefficients')) and (not hasattr(self,'_coefficients')):
            Warning("No coefficients available, call reconstruct() first, return 0")
            return 0
        if hasattr(self,'coefficients'):
            return self.coefficients
        else:
            return self._coefficients

    @property
    def rf(self):
        return self.reconstructed_field

    @property
    def tf(self):
        return self.training_field

    @property
    def mf(self):
        return self.measured_field

    def reconstruct(self, measured_field = None, local_var=False):
        # TODO: make wrapper that adjusts for apertures and frequencies...
        # reconstructing one aperture / frequency at a time and accumulate
        # results before evaluating.
        # keep reconstruction in internal _reconstruct_aperture()
        if measured_field != None: #update field if provided
            self.measured_field = deepcopy(measured_field)
        mf, tf = self.measured_field, self.training_field
        print(''' > Reconstruction using {} + {} '''.format(
            self.decomposition, self.transform))
        if self.decomposition in ['ksvd','oldl','pca','sydl']:
            print(''' > training {} @ {:.1f} Hz, evaluation {} @ {:.1f} Hz'''.format(
                tf.measurement, tf.frequency, mf.measurement, mf.frequency))
        tic = time.time()

        _ = self.Ar # needs to be there before copying and re-created in Monte Carlo
        _ = mf.sidx # sample to maintain the copying

        # All padding should move to soundfield.py, 
        # also reconstruction from patches.
        # if (self.decomposition in ['lpwe'] or 'csc' in self.transform) & (mf.b_single_patch is False):
        if ('csc' in self.transform) & (mf.b_single_patch is False):
            mf.pad_method = 'zero'
            checkdelattr(mf,'_pidx')
            _=mf.pidx

        pn = mf.padlen
        self.pad = lambda x: np.pad(x, pn, mode='constant') if pn else x
        self.crop = lambda x: x[...,pn:-pn, pn:-pn] if pn else x

        split = (hasattr(self,'b_split_complex_fields') and self.b_split_complex_fields)
        self.cstack   = lambda x: np.dstack([np.real(x), np.imag(x)]) if split else x
        self.rstack   = lambda x: np.dstack([x,x]) if split else x
        self.uncstack = lambda x: x[:,:,0] + 1j*x[:,:,1] if split else x

        self.reconstructed_field = deepcopy(self.measured_field) # copy to reconstruction field
        rf = self.reconstructed_field

        if  self.decomposition in ['lpwe', 'ksvd', 'sinc', 'oldl','sydl', 'pca']: # local
            if '011' in mf.measurement:
                rf.pm = np.zeros(mf.pm.shape, dtype = complex)
                for ii in range(mf._pm.shape[-1]):
                    mf.aperture_idx = ii
                    checkdelattr(mf,'sample_idx')
                    gamma, Y = solve_ip( self.transform, self.Ar, 
                            mf.fspatches[:,:,0].T, self.transform_opts)
                    Y  = Y.T.reshape(-1, *rf.psize)
                    y, y_var = reconstruct_from_patches_2d(Y, rf.paddedshape, return_var = local_var)
                    prec = self.crop(y).ravel() # patches are padded in sf.py
                    rf.pm[:,0,ii] = prec
            else: # 019 classroom, dl paper
                gamma, Y = solve_ip( self.transform, self.Ar, 
                        mf.fspatches[:,:,0].T, self.transform_opts)
                Y        = Y.T.reshape(-1, *rf.psize)
                y, y_var = reconstruct_from_patches_2d(Y, rf.paddedshape, return_var = local_var)
                rf.pm = self.crop(y).ravel()[:,np.newaxis] # patches are padded

        elif self.decomposition in ['gpwe']: # global
            if '011' in mf.measurement:
                rf.pm = np.zeros(mf._pm.shape, dtype = complex)
                rf._um = np.zeros((3,*mf._pm.shape), dtype = complex)
                for ii in range(mf._pm.shape[-1]):
                    gamma, Y = solve_ip(self.transform, self.Ar, mf.fsp, self.transform_opts)
                    rf.pm[:,0,ii] = Y.ravel()
                    rf._um[:,:,0,ii] = self.Aur.dot(gamma)
            else:
                gamma, Y = solve_ip(self.transform, self.Ar, mf.fsp, self.transform_opts)
                rf.pm = Y.ravel()[:,np.newaxis]
                # rf.pm = self.Ar.dot(gamma)
                rf._um = self.Aur.dot(gamma)
        else:
            raise ValueError('''No matchin reconstruction method specified for plane
            wave expansion.''')

        self.rec_time = time.time()-tic
        self._coefficients = gamma

        if hasattr(self, 'rec_timestamp'):
            if self.rec_timestamp == int(time.time()):
                time.sleep(1) # delay for unique timestamp
        self.rec_timestamp = int(time.time())
        self.id = self.title()

        # analyse 
        self.assess_reconstruction()

        # reconstructed sound field statistics
        # if  self.decomposition in ['lpwe', 'ksvd', 'sinc', 'oldl','sydl', 'pca']: # local
            # print("\t > local sample nmse {:.2f} dB".format( self.nmse_sample_local))
            # print("\t > local nmse: {:.2f} dB".format( self.nmse_local))
        print("\t > global sample nmse: {:.2f} dB ({:.0f} points)".format(
            self.nmse_sample_global, mf.spatial_sampling))
        # print("\t > global nmse: {:.2f} dB (full aperture)".format( self.nmse))
        # print("\t > mean p_rms^2 [true {:.2f} / meas {:.2f} / fit {:.2f} / rec {:.2f}]".format( mf.prms2, mf.sprms2, rf.sprms2, rf.prms2,))
        print("\t > coh: {:.2f}, nmse: {:.2f} dB, avg_nz: {:.2f}".format(
            self.spatial_coherence , self.nmse, self.avg_nonzeros))
        print('{:=>50}'.format(' total sec {:.2f}'.format(self.rec_time)))
        return self.reconstructed_field, self.Ar, self.coeffs, self.rec_time


    def assess_reconstruction(self):

        # spatial coherence
        self.spatial_coherence =  spatial_coherence(self.rf.fp, self.mf.fp)

        # cardinality
        if (hasattr(self,'coefficients')) or (hasattr(self,'_coefficients')):
            self.__dict__.update( { "avg_nonzeros": assess_sparsity(self.coeffs), })
        else:
            print('no coeffs, skip cardinality update')

        # coefficient norm
        self.coeffnorm = np.linalg.norm(self.coeffs)

        # global residual mse of reconstruction at measurements
        self.nmse_sample_global = nmse(self.rf.fp[self.mf.sidx], self.mf.fp[self.mf.sidx])
        # comparing all apertures - direct access to pm hardcoded
        self.nmse =  nmse(self.rf.fp, self.mf.fp)

    def minimize_hdf_storage(self, full = False):
        self.rf._pm = self.rf.pm
        [checkmoveattr(self.rf.__dict__,key,"_"+key) for key in ["pm","um","pm_var","coefficients"]]

    def clear_reconstruction(self, full = False):
        [checkdelattr(self,key) for key in
                ["id","nmse","nmse_local","nmse_sample_global",
                    "nmse_sample_local","avg_nonzeros","spatial_coherence",
                    "coefficients","_cofficients","A_rec", "_A_rec",
                    "rec_timestamp","reconstructed_field",
                    "_cbpdn"]]
        if full:
            [checkdelattr(self,key) for key in ["A_rec","_A_rec"]]


def read_sfr_from_hdf(filename = './autosave', identifiers = None):
    """
    reads from HDF and casts as objects of a reference class
    identifiers can be one identifier, or a list of identifiers for the highest
    level in the hdf file.
    if identifier is none, all objects are loaded and concatenated to a list

    checks for compatibility with old classes and updates (or even repacks) the stored data if needed.
    """
    import h5py as hdf
    from read_mc import analysis

    if (identifiers is not None) and (type(identifiers) is not list):
        identifiers = [identifiers]

    data = read_from_hdf(filename, identifiers) # if none, read all

    if (identifiers is not None) and (len(identifiers) == 1):
        # cast a single reconstruction object with sound fields
        sfr = SoundfieldReconstruction()
        sfr.__dict__ = deepcopy(data)

        mf  = Soundfield(rdim = np.array([[0,1],[0,1],[0,1]]))
        for fieldkey in ['training_field','measured_field','reconstructed_field']:
            mf.__dict__  = data[fieldkey] # copy to reconstruction field
            sfr.__dict__[fieldkey] = deepcopy(mf)

        return sfr

    else:
        # not specified or several, must be a list of fields
        sfrlist = list()
        keylist = list()
        sfr = SoundfieldReconstruction()
        mf  = Soundfield(rdim = np.array([[0,1],[0,1],[0,1]]))

        # sfrefcfg = default_soundfield_config('/work1/manha/z_data/019_lecture_room.h5')
        # sfref = Soundfield(**sfrefcfg)

        repack = 0 # if data sizes changes, repacking is required to free the disk space
        for key in data.keys():
            # try:
                update = 0
                sfr.__dict__ = data[key]

                for fieldkey in [ 'training_field','measured_field', 'reconstructed_field']:
                        try:
                            mf.__dict__  = data[key][fieldkey] # copy to reconstruction field
                            sfr.__dict__[fieldkey] = deepcopy(mf)
                        except: # legacy
                            if fieldkey == 'training_field':
                                try:
                                    mf.__dict__  = data[key]['learn_field'] # legacy
                                except:
                                    pass
                            elif fieldkey == 'measured_field':
                                mf.__dict__  = data[key]['measured_field'] # legacy
                            sfr.__dict__[fieldkey] = deepcopy(mf)
                            # update = 1

                # compatibility
                if hasattr(sfr.mf, 'loss_factor'):# legacy
                    sfr.mf.spatial_sampling = sfr.mf.loss_factor
                    del sfr.mf.loss_factor
                    update = 2
                if hasattr(sfr, 'avg_cardinality'):# legacy
                    sfr.avg_nonzeros = sfr.avg_cardinality
                    del sfr.avg_cardinality
                    update = 2
                if (key != sfr.title()):
                    with hdf.File(filename, 'r+') as f:
                        f.move(key,sfr.title())
                    key = sfr.title()
                if (sfr.id != sfr.title()):
                    analysis(sfr, b_saveplots = False, b_showplots = False) # checks again for keys
                    sfr.id = sfr.title()
                    update = 3
                elif not np.any([hasattr(sfr,key) for key in ['nmse','spatial_coherence','avg_nonzeros']]):
                    sfr.mf._gen_field()
                    sfr.assess_reconstruction()
                    update = 4
                if hasattr(sfr,'coefficients'): # remove overly large coeffss from lpwe, cpwe
                    # if np.prod(sfr.coeffs.shape) > 1e5:
                    delattr(sfr,'coefficients')
                    update, repack  = 5, 1
                if hasattr(sfr, 'mac'): # legacy
                    sfr.spatial_coherence = sfr.mac
                    del sfr.mac
                    update = 6
                # update data
                if update:
                    print("read_sfr_from_hdf(): update group for criteria nr",update)
                    write_to_hdf(sfr, key, filename)

                # sfr.mf._pm = sfref._pm
                # sfr.assess_reconstruction()

                keylist.append(key)
                sfrlist.append(deepcopy(sfr))
            # except:
                # Warning("No proper sound field reconstruction found, skip {}".format(key))
        # [write_to_hdf(sfr,key,filename) for sfr,key in zip(sfrlist,keylist)]

        if repack: 
            """repack (flag set if significant memory can be saved,
            reloads the complete file"""
            _h5repack(filename)

        return sfrlist

def default_reconstruction_config(decomposition, **kwargs):
    config = dict({
        'decomposition' : decomposition,
        '_figno'        : None,
        })
    if 'gpwe' == decomposition:
        config.update({
            'tag'                     : 'gPWE',
            'nof_components'          : 1000, # samples of k_vectors on the k sphere
            # 'transform'               : 'global_lasso',
            # 'transform'               : 'ridge_cvx_lcurve',
            'transform'               : 'ridge_xval',
            'transform_opts'          : {
                'reg_lambda'          : .01,
                'reg_tolerance'       : 0, # orthogonal base, less noise
                'reg_tol_sig2'        : 1e0,
                'omp_n_nonzero_coefs' : None,
                },
            })
        config['transform'] = 'global_' + config['transform']
    else:
        if 'lpwe' in decomposition:
            config.update({
                'tag'                     : 'pb-PWE',
                # 'nof_components'          : 64, # 1 lambda
                # 'transform'               : 'ridge_xval',
                'nof_components'          : 100, # 1 lambda
                'transform'               : 'lasso_lars_xval',
                'transform_opts'          : {
                    'reg_lambda'          : 0.01,
                    'reg_tolerance'       : 1e-3,
                    'n_iter'              : 30,
                    'reg_tol_sig2'        : 1e0,
                    'omp_n_nonzero_coefs' : None,
                    },
                })
        elif 'oldl' in decomposition:
            config.update({
                'tag'                     : 'OLDL',
                'alpha'                   : 8,
                'nof_components'          : 150,
                'n_iter'                  : 2000, #ensuring convergence
                'batch_size'              : 5,  # structure in DL
                'fit_algorithm'           : 'lars',

                'transform'               : 'lasso_lars_xval', # splits RE/IM
                'transform_opts'          : {
                    'reg_lambda'          : 1e-2,
                    'reg_tolerance'       : 1e-2,
                    'reg_tol_sig2'        : 1e0,
                    'omp_n_nonzero_coefs' : None,
                    },
                })
        elif 'sydl' in decomposition:
             # inherit from normla dl
            config, transopts = default_reconstruction_config('oldl')
            config.update({'transform_opts':transopts,
                    'decomposition': 'sydl',
                    'tag': 'SYDL',
                    'alpha' : 2,
                    })
        elif 'sinc' in decomposition:
            config.update({
                'tag'                     : 'SINC',
                'nof_components'          : 150,
                'transform'               : 'lasso_lars_xval',
                # 'transform'               : 'omp', # uses tolerance (or nonzeros)
                'transform_opts'          : {
                    'reg_lambda'          : 1e-2,
                    'reg_tolerance'       : 1e-2,
                    'reg_tol_sig2'        : 1e0,
                    'omp_n_nonzero_coefs' : None,
                    },
                })
        elif 'pca' in decomposition:
            config.update({
                'tag'                     : 'PCA',
                # 'nof_components'          : 8, # 92.2% variance
                # 'transform'               : 'orthogonal_projection',
                # 'nof_components'          : .98, 18    # TRUNCATED TO 98% variance
                'nof_components'          : 18, # 92.2% variance
                'transform'               : 'ridge_xval',
                # 'transform'               : 'lasso_lars_xval', #sparse alternative, SPLITS RE/IM
                'transform_opts'          : {
                    'reg_lambda'          : .02,
                    'reg_tolerance'       : 1e-1,
                    'reg_tol_sig2'        : 1e0,
                    'omp_n_nonzero_coefs' : None,
                    },
                })
        # config['transform'] = 'local_' + config['transform'] # lets try to delete this, mha 1st june 22

    # update with non-default kwargs
    for key in ['reg_tol_sig2',
            'reg_tolerance'     ,'omp_n_nonzero_coefs',
            'reg_lambda','reg_mu']:
        if key in kwargs:
            config['transform_opts'].update({key:kwargs.pop(key)})
    config.update(kwargs)

    otrans = config.pop('transform_opts')
    return config, otrans

if __name__ == '__main__' :
    from read_mc import analysis


    valopts = default_soundfield_config('019')
    valopts.update({
        'frequency'        : 1000,
        # 'loss_mode' : 'grid',
        'spatial_sampling' : -4.0, 
        'seed' : 192384734,
        'patch_size_in_lambda' : 1.0,
        })

    # recopts, transopts  = default_reconstruction_config('gpwe')
    # recopts, transopts  = default_reconstruction_config('oldl')
    recopts, transopts  = default_reconstruction_config('pca')
    # recopts, transopts  = default_reconstruction_config('lpwe')
    # recopts, transopts  = default_reconstruction_config('sydl')

    trainopts = default_soundfield_config('011')
    # # multifreq
    # frange =  10 # training_freq \pm frange
    # deltaf = 1.0 # in steps of deltaf
    # fc = 600
    # recopts['training_frequency'] = fc #multifreq learnin
    # trainopts['frequency'] = np.arange(fc-frange,fc+frange+1e-4,deltaf)

    # load or generate learning data, load objects
    sfrec_obj  = SoundfieldReconstruction(
            training_field = Soundfield(**trainopts),
            measured_field = Soundfield(**valopts),
            transform_opts = transopts,
            **recopts)
    # sfrec_obj.measured_field = Soundfield(**valopts)
    sfrec_obj.fit()
    sfrec_obj.reconstruct()

    # post processing
    analysis(sfrec_obj, b_showplots = 0, b_singlefig = 2, b_saveplots = False)
    plt.show()
