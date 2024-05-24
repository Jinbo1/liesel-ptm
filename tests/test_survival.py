from unittest.mock import MagicMock

import jax.numpy as jnp
import numpy as np

from liesel_ptm.survival import TransformationDist


# Test for TransformationDist
def test_initialization():
    # Mock dependencies
    transformed_variable = MagicMock()
    transformation_derivative = MagicMock()
    refdist = MagicMock()

    # Instance creation
    trans_dist = TransformationDist(
        transformed_variable=transformed_variable,
        transformation_derivative=transformation_derivative,
        refdist=refdist,
        _name="test_dist",
        _needs_seed=True,
    )

    # Assertions to check if initialization is correct
    assert trans_dist._name == "test_dist"
    assert trans_dist._needs_seed == True
    assert trans_dist.refdist == refdist
    assert trans_dist._per_obs == True


def test_log_prob_property():
    trans_dist = TransformationDist(
        transformed_variable=MagicMock(),
        transformation_derivative=MagicMock(),
        refdist=MagicMock(),
    )
    # Mocking internal state
    trans_dist._value = jnp.array(10.0)

    assert trans_dist.log_prob == jnp.array(10.0)


def test_per_obs_property():
    trans_dist = TransformationDist(
        transformed_variable=MagicMock(),
        transformation_derivative=MagicMock(),
        refdist=MagicMock(),
    )
    # Test setter/getter
    assert trans_dist.per_obs == True
    trans_dist.per_obs = False
    assert trans_dist.per_obs == False


def test_update():
    transformed_variable = MagicMock()
    transformed_variable.value = jnp.array([1.0, 2.0])

    transformation_derivative = MagicMock()
    transformation_derivative.value = jnp.array([0.1, 0.2])

    refdist = MagicMock()
    refdist.log_prob.return_value = jnp.array([-1.0, -2.0])

    trans_dist = TransformationDist(
        transformed_variable=transformed_variable,
        transformation_derivative=transformation_derivative,
        refdist=refdist,
    )

    # Call update method
    result = trans_dist.update()

    # Expected log probability calculation
    expected_log_prob = jnp.array([-1.0, -2.0]) + jnp.log(jnp.array([0.1, 0.2]))
    np.testing.assert_array_almost_equal(result.log_prob, expected_log_prob)

    # Test summing up log probabilities
    trans_dist.per_obs = False
    result = trans_dist.update()
    assert result.log_prob == jnp.sum(expected_log_prob)


import jax.numpy as jnp

##############################################################################
import pytest
from liesel.model import Calc
from liesel.model.nodes import no_model_setter

from liesel_ptm.survival import (
    TransformationDist,  # Importing the class from the module where it's defined
)


class MockDistribution:
    """A mock distribution for testing purposes."""

    def log_prob(self, value):
        return -0.5 * jnp.log(2 * jnp.pi) - 0.5 * value**2  # Standard normal log PDF

    def cdf(self, value):
        return 0.5 * (1.0 + jnp.tanh(value))  # Approximation of standard normal CDF


@pytest.fixture
def setup_transformation_dist():
    """Fixture to set up a TransformationDist instance for testing."""
    transformed_variable = Calc(lambda: jnp.array([0.5, 1.0, -1.5, -2.0]))
    transformation_derivative = Calc(lambda: jnp.array([1.0, 0.8, 1.2, 1.0]))
    refdist = MockDistribution()

    return TransformationDist(
        transformed_variable=transformed_variable,
        transformation_derivative=transformation_derivative,
        refdist=refdist,
    )


def test_log_prob(setup_transformation_dist):
    """Test the log_prob property."""
    node = setup_transformation_dist
    node.update()  # Ensure the update method is called before accessing log_prob
    assert node.log_prob is not None, "Log probability should not be None"


def test_update(setup_transformation_dist):
    """Test the update method."""
    node = setup_transformation_dist
    node.update()

    transformed_variable = node._transformed_variable.value
    deriv = node._transformation_derivative.value

    base_log_prob = node.refdist.log_prob(transformed_variable)
    log_prob_adjustment = jnp.log(jnp.maximum(deriv, 1e-30))
    expected_log_prob = base_log_prob + log_prob_adjustment

    if not node.per_obs:
        expected_log_prob = expected_log_prob.sum()

    assert jnp.allclose(
        node.value, expected_log_prob
    ), "Update method log prob does not match expected value"


def test_per_obs_property(setup_transformation_dist):
    """Test the per_obs property."""
    node = setup_transformation_dist

    # Test getter
    assert node.per_obs is True, "Initial per_obs should be True"

    # Test setter
    node.per_obs = False
    assert node.per_obs is False, "per_obs should be False after setting to False"

    node.per_obs = True
    assert node.per_obs is True, "per_obs should be True after setting to True"


##############################################################################

import jax.numpy as jnp
import pytest
from liesel.model import Calc
from liesel.model.nodes import no_model_setter

from liesel_ptm.survival import (
    TransformationDist_sur,  # Importing the class from the survival module
)


class MockDistribution:
    """A mock distribution for testing purposes."""

    def log_prob(self, value):
        return -0.5 * jnp.log(2 * jnp.pi) - 0.5 * value**2  # Standard normal log PDF

    def cdf(self, value):
        return 0.5 * (1.0 + jnp.tanh(value))  # Approximation of standard normal CDF


@pytest.fixture
def setup_transformation_dist_sur():
    """Fixture to set up a TransformationDist_sur instance for testing."""
    transformed_variable = Calc(lambda: jnp.array([0.5, 1.0, -1.5, -2.0]))
    transformation_derivative = Calc(lambda: jnp.array([1.0, 0.8, 1.2, 1.0]))
    refdist = MockDistribution()
    censoring_indicator = jnp.array([0, 1, 0, 1])

    return TransformationDist_sur(
        transformed_variable=transformed_variable,
        transformation_derivative=transformation_derivative,
        refdist=refdist,
        censoring_indicator=censoring_indicator,
    )


def test_right_censored_log_prob(setup_transformation_dist_sur):
    """Test the right_censored_log_prob method."""
    node = setup_transformation_dist_sur
    transformed_variable = node._transformed_variable.value
    log_prob = node.right_censored_log_prob(transformed_variable)

    expected_log_prob = jnp.log(1 - node.refdist.cdf(transformed_variable))
    assert jnp.allclose(
        log_prob, expected_log_prob
    ), "Right censored log prob does not match expected value"


def test_uncensored_log_prob(setup_transformation_dist_sur):
    """Test the uncensored_log_prob method."""
    node = setup_transformation_dist_sur
    transformed_variable = node._transformed_variable.value
    deriv = node._transformation_derivative.value
    log_prob = node.uncensored_log_prob(transformed_variable, deriv)

    base_log_prob = node.refdist.log_prob(transformed_variable)
    log_prob_adjustment = jnp.log(jnp.maximum(deriv, 1e-30))
    expected_log_prob = base_log_prob + log_prob_adjustment

    assert jnp.allclose(
        log_prob, expected_log_prob
    ), "Uncensored log prob does not match expected value"


def test_update(setup_transformation_dist_sur):
    """Test the update method."""
    node = setup_transformation_dist_sur
    node.update()

    transformed_variable = node._transformed_variable.value
    deriv = node._transformation_derivative.value

    right_censored_log_prob = node.right_censored_log_prob(transformed_variable)
    uncensored_log_prob = node.uncensored_log_prob(transformed_variable, deriv)
    expected_log_prob = jnp.where(
        node.censoring_indicator == 1, right_censored_log_prob, uncensored_log_prob
    )

    if not node.per_obs:
        expected_log_prob = expected_log_prob.sum()

    assert jnp.allclose(
        node.value, expected_log_prob
    ), "Update method log prob does not match expected value"


import jax.numpy as jnp
import numpy as np

##########################################################################
import pytest
import scipy.stats
import tensorflow_probability.substrates.jax as tfp
from jax import random
from liesel.model import Calc, Dist, GraphBuilder, Var, obs, param

import liesel_ptm as ptm
from liesel_ptm.survival import TransformationDist_sur


@pytest.fixture  # fixture decorator to ensure these codes canrepeatedly run.
def generate_data_and_model():
    # Generate survival data
    key = random.PRNGKey(0)  # generate PRNG keys
    shape = ptm.sample_shape(key, nshape=8).sample  # set shape vector
    ln_y = ptm.PTMLocScaleDataGen(shape=shape, loc_fn=lambda x: x, ncov=1)  # set DGP
    ln_y_sample = ptm.PTMLocScaleDataGen.sample(
        key=key, self=ln_y, nobs=100
    )  # draw ln(y)
    y = np.exp(ln_y_sample["y"])
    x = ln_y_sample["x"].reshape(-1, 1)  # Ensure x has shape (100, 1)

    # Create indicators of censored data
    c0 = tfp.distributions.Bernoulli(probs=0.05).sample(
        100, seed=key
    )  # draw samples via tfp

    # Draw censoring time
    u = scipy.stats.uniform.rvs(
        loc=0.001, scale=y - 0.0001, size=100
    )  # draw 100 samples of censoring time

    y_hat = np.copy(y)
    y_hat[c0 == 1] = u[c0 == 1]  # set y_hat as y with censored data u when c = 1

    # Build the model with Liesel
    b0 = param(0.0, name="b0")
    x = obs(x, name="x_ptm")
    b1 = param(0.0, name="b1")

    def linear_model(x, b0, b1):  # set the linear regression of x
        return b0 + x * b1

    mu = Var(Calc(linear_model, x=x, b0=b0, b1=b1), name="mu")
    sigma_sq = param(10.0, name="sigma_sq")
    sigma = Var(Calc(jnp.sqrt, sigma_sq), name="sigma")
    y_dist = Dist(tfp.distributions.Normal, loc=mu, scale=sigma)
    y_ptm = obs(jnp.log(y_hat), y_dist, name="y_sur")

    # Ensure parameters are added to the graph
    sur_ptm = GraphBuilder().add(b0).add(b1).add(sigma_sq).add(mu).add(sigma).add(y_ptm)
    sur_ptm_model = sur_ptm.build_model()

    return {
        "y": y,
        "x": x,
        "c0": c0,
        "u": u,
        "y_hat": y_hat,
        "mu": mu,
        "sigma": sigma,
        "sur_ptm_model": sur_ptm_model,
    }


def extract_values(var):
    """Extract the value from a Liesel Var node."""
    return var.value


def test_transformation_dist_sur(generate_data_and_model):
    """Test the TransformationDist_sur class within the model context."""
    data = generate_data_and_model
    mu_value = extract_values(data["mu"])
    sigma_value = extract_values(data["sigma"])
    refdist = tfp.distributions.Normal(loc=mu_value, scale=sigma_value)
    censoring_indicator = data["c0"]

    transformation_dist_sur = TransformationDist_sur(
        transformed_variable=Calc(lambda: jnp.log(data["y_hat"])),
        transformation_derivative=Calc(lambda: jnp.ones_like(data["y_hat"])),
        refdist=refdist,
        censoring_indicator=censoring_indicator,
    )

    # Test right censored log prob
    right_censored_log_prob = transformation_dist_sur.right_censored_log_prob(
        jnp.log(data["y_hat"])
    )
    expected_right_censored_log_prob = jnp.log(1 - refdist.cdf(jnp.log(data["y_hat"])))
    assert jnp.allclose(
        right_censored_log_prob, expected_right_censored_log_prob
    ), "Right censored log prob does not match expected value"

    # Test uncensored log prob
    uncensored_log_prob = transformation_dist_sur.uncensored_log_prob(
        jnp.log(data["y_hat"]), jnp.ones_like(data["y_hat"])
    )
    base_log_prob = refdist.log_prob(jnp.log(data["y_hat"]))
    log_prob_adjustment = jnp.log(jnp.maximum(jnp.ones_like(data["y_hat"]), 1e-30))
    expected_uncensored_log_prob = base_log_prob + log_prob_adjustment
    assert jnp.allclose(
        uncensored_log_prob, expected_uncensored_log_prob
    ), "Uncensored log prob does not match expected value"

    # Test update method
    transformation_dist_sur.update()
    combined_log_prob = jnp.where(
        censoring_indicator == 1, right_censored_log_prob, uncensored_log_prob
    )
    if not transformation_dist_sur.per_obs:
        combined_log_prob = combined_log_prob.sum()
    assert jnp.allclose(
        transformation_dist_sur.value, combined_log_prob
    ), "Update method log prob does not match expected value"


@pytest.mark.parametrize("per_obs", [True, False])
def test_per_obs_property(generate_data_and_model, per_obs):
    """Test the per_obs property."""
    data = generate_data_and_model
    mu_value = extract_values(data["mu"])
    sigma_value = extract_values(data["sigma"])
    refdist = tfp.distributions.Normal(loc=mu_value, scale=sigma_value)
    censoring_indicator = data["c0"]

    transformation_dist_sur = TransformationDist_sur(
        transformed_variable=Calc(lambda: jnp.log(data["y_hat"])),
        transformation_derivative=Calc(lambda: jnp.ones_like(data["y_hat"])),
        refdist=refdist,
        censoring_indicator=censoring_indicator,
    )

    # Test getter and setter for per_obs property
    transformation_dist_sur.per_obs = per_obs
    assert (
        transformation_dist_sur.per_obs == per_obs
    ), f"per_obs should be {per_obs} after setting to {per_obs}"

    transformation_dist_sur.update()
    combined_log_prob = jnp.where(
        censoring_indicator == 1,
        transformation_dist_sur.right_censored_log_prob(jnp.log(data["y_hat"])),
        transformation_dist_sur.uncensored_log_prob(
            jnp.log(data["y_hat"]), jnp.ones_like(data["y_hat"])
        ),
    )
    if not per_obs:
        combined_log_prob = combined_log_prob.sum()
    assert jnp.allclose(
        transformation_dist_sur.value, combined_log_prob
    ), f"Update method log prob does not match expected value with per_obs={per_obs}"
