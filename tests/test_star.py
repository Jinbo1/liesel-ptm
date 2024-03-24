from collections.abc import Iterator

import jax
import liesel.goose as gs
import liesel.model as lsl
import numpy as np
import pandas as pd
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd
from scipy import stats

import liesel_ptm as ptm
from liesel_ptm import bsplines
from liesel_ptm import nodes as nd
from liesel_ptm import ptm_ls

np.random.seed(2407)
x = np.random.uniform(low=-1.0, high=1.0, size=300)
sigma = np.exp(-0.3 + 0.2 * x)
mu = 1.2 + 0.4 * x
y = mu + sigma * np.random.normal(size=300)

nparam = 10
Z = nd.sumzero_coef(nparam)
K = Z.T @ nd.diffpen(nparam) @ Z
knots = bsplines.kn(x, n_params=nparam)


def basis_fn(x):
    return bsplines.bspline_basis(x, knots, 3) @ Z


@pytest.fixture
def tau():
    prior = lsl.Dist(tfd.HalfCauchy, loc=0.0, scale=25.0)
    yield nd.ExpParam(10.0, prior, name="tau")


@pytest.fixture(scope="module")
def samples():
    path = "tests/resources/normal_nonlinear_star/results.pickle"
    results = gs.engine.SamplingResults.pkl_load(path)
    yield results.get_posterior_samples()


@pytest.fixture(scope="module")
def model() -> Iterator[ptm_ls.PTMLocScale]:
    df = pd.read_csv("tests/resources/normal_nonlinear_star/data.csv")

    x = df.x.to_numpy()
    y = df.y.to_numpy()

    path = "tests/resources/normal_nonlinear_star/knots.pickle"
    model = ptm_ls.PTMLocScale._from_knots(path, y)

    tau2 = nd.VarHalfCauchy(100.0, scale=25.0, name="tau2")
    star = nd.S.pspline(x, nparam=10, tau2=tau2, name="x_loc")

    model.loc_model += star
    model.log_scale_model += ptm.LinearTerm(x, name="x_scale")
    yield model


@pytest.fixture(scope="module")
def pred(model, samples) -> Iterator[ptm_ls.PTMLocScalePredictions]:
    pred = ptm_ls.PTMLocScalePredictions(samples, model, x_loc=None, x_scale=None)
    yield pred


class TestStructuredAdditiveTerm:
    def test_init(self, tau) -> None:
        tau2 = nd.VarHalfCauchy(100.0, scale=25.0, name="tau2")
        star = nd.S.pspline(x, nparam=20, tau2=tau2, name="x_loc")
        model = lsl.GraphBuilder().add(star.smooth).build_model()

        assert model is not None

    @pytest.mark.skip(reason="I only test this manually at the moment.")
    def test_mcmc(self, tau) -> None:
        model = ptm_ls.PTMLocScale.from_nparam(y, nparam=30)

        tau2 = nd.VarHalfCauchy(100.0, scale=25.0, name="tau2")
        model.loc_model += nd.S.pspline(x, nparam=20, tau2=tau2, name="x_loc")
        model.log_scale_model += ptm.LinearTerm(x, name="x_scale")

        graph = model.build_graph(maxiter=10)

        eb = gs.EngineBuilder(seed=192023, num_chains=4)
        eb = model.setup_engine_builder(eb, graph)
        eb.set_duration(warmup_duration=1000, posterior_duration=1000)
        eb.positions_included += [
            "normalization_tau2_root_transformed",
            "normalization_shape",
        ]
        engine = eb.build()

        engine.sample_all_epochs()
        results = engine.get_results()
        samples = results.get_posterior_samples()

        pred = ptm_ls.PTMLocScalePredictions(samples, model, x_loc=x, x_scale=x)

        z = pred.predict_z().mean(axis=(0, 1))
        _, pval = stats.normaltest(z)
        assert pval > 0.05

        shape = samples["normalization_shape"].mean(axis=(0, 1))
        assert np.allclose(shape, 0.0, atol=0.2)

        loc_intercept = pred.loc_intercept.mean()
        assert loc_intercept == pytest.approx(1.2, abs=0.2)

        scale_intercept = np.log(pred.scale_intercept_exp).mean()
        assert scale_intercept == pytest.approx(-0.3, abs=0.2)

        scale_coef = samples["x_scale_coef"].mean()
        assert scale_coef == pytest.approx(0.2, abs=0.2)

    def test_summarise(self, pred) -> None:
        df = pred.model.loc_model.terms["x_loc"].summarise_by_quantiles(pred.samples)
        assert df.shape == (300, 7)
        assert not df.isnull().values.any()

    def test_samples(self, pred) -> None:
        key = jax.random.PRNGKey(4)
        term = pred.model.loc_model.terms["x_loc"]
        df = term.summarise_by_samples(key, pred.samples, n=10)
        assert df.shape == (3000, 6)
        assert not df.isnull().values.any()
