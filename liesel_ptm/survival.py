from __future__ import annotations
import jax.numpy as jnp
import liesel.model as lsl
from liesel.model.nodes import no_model_setter
from .custom_types import Array, TFPDistribution

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



