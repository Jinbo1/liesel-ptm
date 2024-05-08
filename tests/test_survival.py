from __future__ import annotations
import jax.numpy as jnp
import liesel.model as lsl
from liesel.model.nodes import no_model_setter
#from .custom_types import Array, TFPDistribution
import liesel_ptm as ptm
from jax import random
import scipy.stats
import numpy as np

import liesel_ptm.survival as sur



# Generate survival data (easy case)
key = random.key(0) # generate PRNG keys
shape = ptm.sample_shape(key, nshape=8).sample # set shape vector
x = random.uniform(key) # generate uniformly distributed covariate
ln_y = ptm.PTMLocScaleDataGen(shape = shape,loc_fn= lambda x: x, ncov=1) # set DGP
ln_y_sample = ptm.PTMLocScaleDataGen.sample(key=key, self=ln_y, nobs= 100) # draw ln(y)
#y = jnp.exp(ln_y_sample["y"]) # compute duration times by exp(ln(y))
y = np.exp(ln_y_sample["y"])

# Create indicators of censored data
c = scipy.stats.bernoulli.rvs(p = 0.05, size = 100)

# draw censoring time
u = scipy.stats.uniform.rvs(loc = 0.001, scale = y - 0.0001, size = 100) # draw 100 samples of sencoring time

y_hat = np.copy(y)
y_hat[c ==1] = u[c==1] # set y_hat as y with censored data u when c = 1
    


def test_data_simulation():
    assert 

