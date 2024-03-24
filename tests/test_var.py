import liesel.model as lsl
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd

from liesel_ptm import nodes as nd


class TestVar:
    def test_transform_instance(self) -> None:
        prior = lsl.Dist(tfd.HalfCauchy, loc=0.0, scale=25.0)
        tau = nd.Var(10.0, prior, name="tau")
        log_tau = tau.transform(tfb.Exp())
        tau.update()

        assert tau.weak
        assert not log_tau.weak

        assert tau.value == pytest.approx(np.exp(log_tau.value))
        assert tau.value == pytest.approx(10.0)
        assert log_tau.value == pytest.approx(np.log(10.0))

        prior = lsl.Dist(tfd.HalfCauchy, loc=0.0, scale=25.0)
        tau = nd.Var(10.0, prior, name="tau")
        log_tau_gb = lsl.GraphBuilder().transform(tau, tfb.Exp)

        assert tau.weak
        assert not log_tau.weak

        tau.update()
        assert tau.value == pytest.approx(np.exp(log_tau.value))
        assert tau.value == pytest.approx(10.0)
        assert log_tau_gb.value == pytest.approx(np.log(10.0))

        log_tau.dist_node.update()
        log_tau_gb.dist_node.update()
        assert log_tau.log_prob == pytest.approx(log_tau_gb.log_prob)

    def test_transform_default(self) -> None:
        prior = lsl.Dist(tfd.HalfCauchy, loc=0.0, scale=25.0)
        tau = nd.Var(10.0, prior, name="tau")
        log_tau = tau.transform(None)
        tau.update()

        assert tau.weak
        assert not log_tau.weak

        assert tau.value == pytest.approx(np.exp(log_tau.value))
        assert tau.value == pytest.approx(10.0)
        assert log_tau.value == pytest.approx(np.log(10.0))

        prior = lsl.Dist(tfd.HalfCauchy, loc=0.0, scale=25.0)
        tau = nd.Var(10.0, prior, name="tau")
        log_tau_gb = lsl.GraphBuilder().transform(tau)

        assert tau.weak
        assert not log_tau.weak

        tau.update()
        assert tau.value == pytest.approx(np.exp(log_tau.value))
        assert tau.value == pytest.approx(10.0)
        assert log_tau_gb.value == pytest.approx(np.log(10.0))

        log_tau.dist_node.update()
        log_tau_gb.dist_node.update()
        assert log_tau.log_prob == pytest.approx(log_tau_gb.log_prob)
