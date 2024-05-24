from __future__ import annotations

import jax.numpy as jnp
import liesel.model as lsl
from liesel.model.nodes import no_model_setter

from liesel_ptm.custom_types import Array, KeyArray, TFPDistribution


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


##############################################################################


# Update the calss to fit survival data
class TransformationDist_sur(lsl.Dist):
    """A transformation-distribution node for a conditional transformation model."""

    def __init__(
        self,
        transformed_variable: lsl.Calc,  # transformed variables
        transformation_derivative: lsl.Calc,  # derivative of the transformation function
        refdist: TFPDistribution,  # reference distribution
        censoring_indicator: Array,  # A censoring indicator for ecah observation
        _name: str = "",
        _needs_seed: bool = False,  # optional parameters for naming the node and specifying a seed
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
        self.censoring_indicator = censoring_indicator  # Store censoring type

    @property
    def log_prob(self) -> Array:
        """The log-probability of the distribution."""
        return self.value  # returns the log-probability of the distribution

    @property
    def per_obs(self) -> bool:
        """Whether the log-probability is stored per observation or summed up."""
        return (
            self._per_obs
        )  # indicates whether the log-probability should be stored per observation or summed over all observations

    @per_obs.setter  # sets the `per_obs` property
    @no_model_setter
    def per_obs(self, per_obs: bool):
        self._per_obs = per_obs

    def right_censored_log_prob(self, transformed_variable):
        return jnp.log(
            1 - self.refdist.cdf(transformed_variable)
        )  # Use the complementary CDF for right censoring

    def uncensored_log_prob(
        self, transformed_variable, deriv
    ):  # calculates log-probability for uncensored data
        base_log_prob = self.refdist.log_prob(transformed_variable)
        deriv = jnp.maximum(deriv, 1e-30)
        log_prob_adjustment = jnp.log(deriv)
        return jnp.add(base_log_prob, log_prob_adjustment)

    def update(self) -> TransformationDist_sur:  # computes log-prob seperately
        transformed_variable = self._transformed_variable.value
        deriv = self._transformation_derivative.value

        # Calculate log-probabilities for censored and uncensored data
        right_censored_log_prob = self.right_censored_log_prob(transformed_variable)
        uncensored_log_prob = self.uncensored_log_prob(transformed_variable, deriv)

        # Combine log-probabilities based on the censoring indicator
        log_prob = jnp.where(
            self.censoring_indicator == 1, right_censored_log_prob, uncensored_log_prob
        )  # select approperate log-prob based on censoring indicator

        if not self.per_obs and hasattr(log_prob, "sum"):
            log_prob = log_prob.sum()  # sums the log-prob if `per_obs` is `False`

        self._value = log_prob
        self._outdated = False
        return self
