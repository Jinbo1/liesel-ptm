from __future__ import annotations
import jax.numpy as jnp
import liesel.model as lsl
from liesel.model.nodes import no_model_setter
from .custom_types import Array, TFPDistribution
import liesel_ptm as ptm
from jax import random
import scipy.stats
import numpy as np



class SurvivalTransformationDist(lsl.Dist):
    "A transformation-distribution node for a survival likelihood"


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
    





class TransDistSur(lsl.Dist):
    def __init__(
            self,
            transformed_variable: lsl.Calc
    )

class TransformationDist(lsl.Dist):
    """A transformation-distribution node for a conditional transformation model."""

    def __init__(
        self,
        transformed_variable: lsl.Calc,
        transformation_derivative: lsl.Calc,
        refdist: TFPDistribution,
        _name: str = "",
        _needs_seed: bool = False,
    ):
        super(lsl.Dist, self).__init__(
            transformed_variable,
            transformation_derivative,
            _name=_name,
            _needs_seed=_needs_seed,
        )

        self._per_obs = True
        self.refdist = refdist
        self._transformed_variable = transformed_variable
        self._transformation_derivative = transformation_derivative

    @property
    def log_prob(self) -> Array:
        """The log-probability of the distribution."""
        return self.value

    @property
    def per_obs(self) -> bool:
        """Whether the log-probability is stored per observation or summed up."""
        return self._per_obs

    @per_obs.setter
    @no_model_setter
    def per_obs(self, per_obs: bool):
        self._per_obs = per_obs

    def update(self) -> TransformationDist:
        base_log_prob = self.refdist.log_prob(self._transformed_variable.value)
        deriv = self._transformation_derivative.value
        deriv = jnp.maximum(deriv, 1e-30)
        log_prob_adjustment = jnp.log(deriv)
        log_prob = jnp.add(base_log_prob, log_prob_adjustment)

        if not self.per_obs and hasattr(log_prob, "sum"):
            log_prob = log_prob.sum()

        self._value = log_prob
        self._outdated = False
        return self

