import numpy as np

def random_phase(N = 1,seed = None, block_reset = False):
    """!
    @param *N: number of samples, default one
    @param *seed: set to force random number seed
    @param *block_reset: block seed set

    @return phases: uniformly distributed random phase vector with unit amplitude
    """
    # could also used np.random.set_state to reset to e.g. initial state
    if not block_reset:
        np.random.seed(seed)
    phases = np.exp(2j*np.pi*np.random.uniform(0,1,N))
    return phases
