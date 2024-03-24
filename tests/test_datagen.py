import jax
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

from liesel_ptm import datagen as dg
from liesel_ptm.datagen import sample_shape

# from plotnine import *


key = jax.random.PRNGKey(1723)


class TestDGP:
    def test_sample_cov(self) -> None:
        x1 = dg.DataGenerator.sample_covariates(key, 10, 1)
        assert x1.shape == (10,)

        x2 = dg.DataGenerator.sample_covariates(key, 14, 2)
        assert x2.shape == (14, 2)

    def test_sample_z(self) -> None:
        z1 = dg.DataGenerator.sample_z(key, 10)
        assert z1.shape == (10,)

        z2 = dg.DataGenerator.sample_z(key, 14)
        assert z2.shape == (14,)

    def test_cov1(self) -> None:
        x = dg.DataGenerator.sample_covariates(key, 10, 1)
        fx = dg.DataGenerator.cov1_linear(x)
        assert x == pytest.approx(fx)

    def test_cov2(self) -> None:
        x = dg.DataGenerator.sample_covariates(key, 10, 1)
        fx = dg.DataGenerator.cov2_ushaped(x)
        assert x != pytest.approx(fx)
        assert x.shape == fx.shape

    def test_cov3(self) -> None:
        x = dg.DataGenerator.sample_covariates(key, 10, 1)
        fx = dg.DataGenerator.cov3_oscillating(x)
        assert x != pytest.approx(fx)
        assert x.shape == fx.shape

    def test_cov4(self) -> None:
        x = dg.DataGenerator.sample_covariates(key, 10, 1)
        fx = dg.DataGenerator.cov4_bell(x)
        assert x != pytest.approx(fx)
        assert x.shape == fx.shape


class TestExpDGPBase:
    def test_sample(self) -> None:
        dgp = dg.ExponentialDataGen()
        data = dgp.sample(key, 20)

        assert data["y"].shape == (20,)

    def test_transformation(self) -> None:
        dgp = dg.ExponentialDataGen()
        data = dgp.sample(key, 20)
        y = data["y"]
        z = data["z"]
        x = data["x0"]
        assert dgp.transformation(y, x) == pytest.approx(z, abs=1e-4)
        assert dgp.transformation_inverse(z, x) == pytest.approx(y, abs=1e-4)

    def test_cdf(self) -> None:
        dgp = dg.ExponentialDataGen()
        data = dgp.sample(key, 20)
        cdf = dgp.cdf(data["y"], data["x0"])
        assert cdf.shape == (20,)

    def test_pdf(self) -> None:
        dgp = dg.ExponentialDataGen()

        ynew = np.linspace(0.1, 2.0, 100)
        xnew = 1.0
        pdf = dgp.pdf(ynew, xnew)
        assert pdf.shape == (100,)

    def test_quantile(self) -> None:
        dgp = dg.ExponentialDataGen()

        xnew = 1.0
        y = dgp.quantile(0.1, xnew)
        assert y is not None

        y2 = dgp.quantile(0.9, xnew)
        assert y2 > y

        y3 = dgp.quantile(np.linspace(0.1, 0.9, 10), xnew)
        assert np.all(np.diff(y3) > 0.0)


class TestExpLogitDGPBase:
    def test_sample(self) -> None:
        dgp = dg.ExpLogitDGPBase()
        data = dgp.sample(key, 20)

        assert data["y"].shape == (20,)

    def test_transformation(self) -> None:
        dgp = dg.ExpLogitDGPBase()
        data = dgp.sample(key, 200)
        y = data["y"]
        z = data["z"]
        x = data["x0"]
        assert dgp.transformation(y, x) == pytest.approx(z, abs=1e-4)
        assert dgp.transformation_inverse(z, x) == pytest.approx(y, abs=1e-4)

        # qplot(y, dgp.h(y, 1.0))

    def test_cdf(self) -> None:
        dgp = dg.ExpLogitDGPBase()
        data = dgp.sample(key, 20)
        cdf = dgp.cdf(data["y"], data["x0"])
        assert cdf.shape == (20,)

    def test_pdf(self) -> None:
        dgp = dg.ExpLogitDGPBase()

        ynew = np.linspace(0.1, 10.0, 100)
        xnew = 1.0
        pdf = dgp.pdf(ynew, xnew)
        assert pdf.shape == (100,)

    def test_quantile(self) -> None:
        dgp = dg.ExpLogitDGPBase()

        xnew = 1.0
        y = dgp.quantile(0.1, xnew)
        assert y is not None

        y2 = dgp.quantile(0.9, xnew)
        assert y2 > y

        y3 = dgp.quantile(np.linspace(0.1, 0.9, 10), xnew)
        assert np.all(np.diff(y3) > 0.0)


class TestLLSTMDGP:
    def test_sample(self) -> None:
        key = jax.random.PRNGKey(1848)
        k1, k2 = jax.random.split(key)
        shape = sample_shape(k1, 20).sample
        dgp = dg.TAMLocScaleDataGen(-4.0, 4.0, shape)
        data = dgp.sample(k2, 200)

        assert data["y"].shape == (200,)

    def test_transformation(self) -> None:
        key = jax.random.PRNGKey(1848)
        k1, k2 = jax.random.split(key)
        shape = sample_shape(k1, 20).sample
        dgp = dg.TAMLocScaleDataGen(-4.0, 4.0, shape)
        data = dgp.sample(k2, 200)
        y = data["y"]
        z = data["z"]
        assert dgp.transformation(y) == pytest.approx(z, abs=1e-4)
        assert dgp.transformation_inverse(z) == pytest.approx(y, abs=1e-4)

    def test_cdf(self) -> None:
        key = jax.random.PRNGKey(1848)
        k1, k2 = jax.random.split(key)
        shape = sample_shape(k1, 20).sample
        dgp = dg.TAMLocScaleDataGen(-4.0, 4.0, shape)
        data = dgp.sample(k2, 200)
        cdf = dgp.cdf(data["y"])
        assert cdf.shape == (200,)

    def test_pdf(self) -> None:
        key = jax.random.PRNGKey(1848)
        k1, _ = jax.random.split(key)
        shape = sample_shape(k1, 20).sample
        dgp = dg.TAMLocScaleDataGen(-4.0, 4.0, shape)

        ynew = np.linspace(-4.0, 4.0, 100)
        pdf = dgp.pdf(ynew)
        assert pdf.shape == (100,)

    def test_quantile(self) -> None:
        key = jax.random.PRNGKey(1848)
        k1, _ = jax.random.split(key)
        shape = sample_shape(k1, 20).sample
        dgp = dg.TAMLocScaleDataGen(-4.0, 4.0, shape)

        xnew = 1.0
        y = dgp.quantile(0.1, xnew)
        assert y is not None

        y2 = dgp.quantile(0.9, xnew)
        assert y2 > y

        y3 = dgp.quantile(np.linspace(0.1, 0.9, 10), xnew)
        assert np.all(np.diff(y3) > 0.0)

    def test_loc_fn(self) -> None:
        key = jax.random.PRNGKey(1848)
        k1, k2, k3 = jax.random.split(key, 3)
        shape = sample_shape(k1, 20).sample

        b = jax.random.normal(k2, shape=(3,))
        x = jax.random.uniform(k3, shape=(40, 3))

        def _locfn(x):
            return x @ b

        dgp = dg.TAMLocScaleDataGen(-4.0, 4.0, shape, loc_fn=_locfn)

        assert np.allclose(dgp.loc(x), x @ b)

    def test_scale_fn_negative_scale_fails(self) -> None:
        key = jax.random.PRNGKey(1848)
        k1, k2, k3 = jax.random.split(key, 3)
        shape = sample_shape(k1, 20).sample

        b = jax.random.normal(k2, shape=(3,))
        x = jax.random.uniform(k3, shape=(40, 3))

        def _scalefn(x):
            return x @ b

        dgp = dg.TAMLocScaleDataGen(-4.0, 4.0, shape, scale_fn=_scalefn)

        with pytest.raises(AssertionError):
            dgp.scale(x)

    def test_scale_fn(self) -> None:
        key = jax.random.PRNGKey(1848)
        k1, k2, k3 = jax.random.split(key, 3)
        shape = sample_shape(k1, 20).sample

        b = jax.random.normal(k2, shape=(3,))
        x = jax.random.uniform(k3, shape=(40, 3))

        def _scalefn(x):
            return np.exp(x @ b)

        dgp = dg.TAMLocScaleDataGen(-4.0, 4.0, shape, scale_fn=_scalefn)

        assert np.allclose(dgp.scale(x), np.exp(x @ b))

    def test_sample_with_covariates(self) -> None:
        key = jax.random.PRNGKey(1848)
        k1, k2, _, k4 = jax.random.split(key, 4)
        shape = sample_shape(k1, 20).sample

        b = jax.random.normal(k2, shape=(3,))

        def _locfn(x):
            return x @ b

        dgp = dg.TAMLocScaleDataGen(-4.0, 4.0, shape, loc_fn=_locfn, ncov=3)

        data = dgp.sample(k4, 60)
        df = dgp.to_df(data)

        assert df.shape == (60, 11)

    def test_reduction_to_normal(self) -> None:
        key = jax.random.PRNGKey(1848)
        _, k2, _, k4 = jax.random.split(key, 4)
        shape = np.zeros(10)

        b = jax.random.normal(k2, shape=(3,))

        def _locfn(x):
            return x @ b

        dgp = dg.TAMLocScaleDataGen(-4.0, 4.0, shape, loc_fn=_locfn, ncov=3)

        data = dgp.sample(k4, 60)
        x = data["x"]
        dist = tfd.Normal(loc=dgp.loc(x), scale=dgp.scale(x))
        ref_log_prob = dist.log_prob(data["y"])

        assert np.allclose(ref_log_prob, data["log_prob"])

    def test_nonnormal(self) -> None:
        key = jax.random.PRNGKey(1848)
        k1, k2, _, k4 = jax.random.split(key, 4)
        shape = sample_shape(k1, 20).sample

        b = jax.random.normal(k2, shape=(3,))

        def _locfn(x):
            return x @ b

        dgp = dg.TAMLocScaleDataGen(-4.0, 4.0, shape, loc_fn=_locfn, ncov=3)

        data = dgp.sample(k4, 60)
        x = data["x"]
        dist = tfd.Normal(loc=dgp.loc(x), scale=dgp.scale(x))
        ref_log_prob = dist.log_prob(data["y"])

        assert not np.allclose(ref_log_prob, data["log_prob"])

    def test_to_df_no_covariate(self) -> None:
        key = jax.random.PRNGKey(1848)
        k1, _, _, k4 = jax.random.split(key, 4)
        shape = sample_shape(k1, 20).sample

        dgp = dg.TAMLocScaleDataGen(-4.0, 4.0, shape)

        data = dgp.sample(k4, 60)
        df = dgp.to_df(data)

        assert df.shape == (60, 8)


class TestLocScale:
    def test_sample(self) -> None:
        key = jax.random.PRNGKey(1848)
        k1, k2 = jax.random.split(key)
        shape = sample_shape(k1, 20).sample
        dgp = dg.PTMLocScaleDataGen(shape)
        data = dgp.sample(k2, 200)

        assert data["y"].shape == (200,)

    def test_transformation(self) -> None:
        key = jax.random.PRNGKey(1848)
        k1, k2 = jax.random.split(key)
        shape = sample_shape(k1, 20).sample
        dgp = dg.PTMLocScaleDataGen(shape)
        data = dgp.sample(k2, 200)
        y = data["y"]
        z = data["z"]
        assert dgp.transformation(y) == pytest.approx(z, abs=1e-4)
        assert dgp.transformation_inverse(z) == pytest.approx(y, abs=1e-4)

    def test_cdf(self) -> None:
        key = jax.random.PRNGKey(1848)
        k1, k2 = jax.random.split(key)
        shape = sample_shape(k1, 20).sample
        dgp = dg.PTMLocScaleDataGen(shape)
        data = dgp.sample(k2, 200)
        cdf = dgp.cdf(data["y"])
        assert cdf.shape == (200,)

    def test_pdf(self) -> None:
        key = jax.random.PRNGKey(1848)
        k1, _ = jax.random.split(key)
        shape = sample_shape(k1, 20).sample
        dgp = dg.PTMLocScaleDataGen(shape)

        ynew = np.linspace(-4.0, 4.0, 100)
        pdf = dgp.pdf(ynew)
        assert pdf.shape == (100,)

    def test_quantile(self) -> None:
        key = jax.random.PRNGKey(1848)
        k1, _ = jax.random.split(key)
        shape = sample_shape(k1, 20).sample
        dgp = dg.PTMLocScaleDataGen(shape)

        xnew = np.atleast_1d(1.0)
        y = dgp.quantile(0.1, xnew)
        assert y is not None

        y2 = dgp.quantile(0.9, xnew)
        assert y2 > y

        y3 = dgp.quantile(np.linspace(0.1, 0.9, 10), xnew)
        assert np.all(np.diff(y3) > 0.0)

    def test_loc_fn(self) -> None:
        key = jax.random.PRNGKey(1848)
        k1, k2, k3 = jax.random.split(key, 3)
        shape = sample_shape(k1, 20).sample

        b = jax.random.normal(k2, shape=(3,))
        x = jax.random.uniform(k3, shape=(40, 3))

        def _locfn(x):
            return x @ b

        dgp = dg.PTMLocScaleDataGen(shape, loc_fn=_locfn)

        assert np.allclose(dgp.loc(x), x @ b)

    def test_scale_fn_negative_scale_fails(self) -> None:
        key = jax.random.PRNGKey(1848)
        k1, k2, k3 = jax.random.split(key, 3)
        shape = sample_shape(k1, 20).sample

        b = jax.random.normal(k2, shape=(3,))
        x = jax.random.uniform(k3, shape=(40, 3))

        def _scalefn(x):
            return x @ b

        dgp = dg.PTMLocScaleDataGen(shape, scale_fn=_scalefn)

        with pytest.raises(AssertionError):
            dgp.scale(x)

    def test_scale_fn(self) -> None:
        key = jax.random.PRNGKey(1848)
        k1, k2, k3 = jax.random.split(key, 3)
        shape = sample_shape(k1, 20).sample

        b = jax.random.normal(k2, shape=(3,))
        x = jax.random.uniform(k3, shape=(40, 3))

        def _scalefn(x):
            return np.exp(x @ b)

        dgp = dg.PTMLocScaleDataGen(shape, scale_fn=_scalefn)

        assert np.allclose(dgp.scale(x), np.exp(x @ b))

    def test_sample_with_covariates(self) -> None:
        key = jax.random.PRNGKey(1848)
        k1, k2, _, k4 = jax.random.split(key, 4)
        shape = sample_shape(k1, 20).sample

        b = jax.random.normal(k2, shape=(3,))

        def _locfn(x):
            return x @ b

        dgp = dg.PTMLocScaleDataGen(shape, loc_fn=_locfn, ncov=3)

        data = dgp.sample(k4, 60)
        df = dgp.to_df(data)

        assert df.shape == (60, 14)

    def test_normalization_mean_and_scale(self) -> None:
        key = jax.random.PRNGKey(1848)
        k1, k2, _, k4 = jax.random.split(key, 4)
        shape = sample_shape(k1, 20).sample
        dgp = dg.PTMLocScaleDataGen(shape)
        q = tfd.Normal(loc=0.0, scale=1.0).quantile(np.linspace(0.001, 0.999, 300))

        zt = dgp.normalization_inv(q)
        z = dgp.normalization(zt)

        assert z.mean() == pytest.approx(0.0, abs=1e-1)
        assert z.std() == pytest.approx(1.0, abs=1e-1)
        assert np.allclose(q, z, atol=1e-4)

    def test_sample_mean_and_scale(self) -> None:
        key = jax.random.PRNGKey(1849)
        k1, k2, _, k4 = jax.random.split(key, 4)
        shape = sample_shape(k1, 20).sample
        dgp = dg.PTMLocScaleDataGen(shape)

        data = dgp.sample(k4, 2000)

        y = data["y"]

        assert y.mean() == pytest.approx(0.0, abs=1e-1)
        assert y.std() == pytest.approx(1.0, abs=1e-1)

    def test_normalization_normal(self) -> None:
        key = jax.random.PRNGKey(1848)
        _, k2, _, k4 = jax.random.split(key, 4)
        shape = np.zeros(10)

        dgp = dg.PTMLocScaleDataGen(shape)

        y = tfd.Normal(loc=0.0, scale=1.0).quantile(np.linspace(0.01, 0.99, 100))
        z = dgp.transformation(y)

        assert np.allclose(y, z, atol=1e-3)
        # (
        #     ggplot()
        #     + geom_point(aes(y, z))
        #     + geom_abline(slope=1, intercept=0, color="red")
        # )

    def test_reduction_to_normal(self) -> None:
        key = jax.random.PRNGKey(1848)
        _, k2, _, k4 = jax.random.split(key, 4)
        shape = np.zeros(10)

        b = jax.random.normal(k2, shape=(3,))

        def _locfn(x):
            return x @ b

        dgp = dg.PTMLocScaleDataGen(shape, loc_fn=_locfn, ncov=3)

        data = dgp.sample(k4, 60)
        x = data["x"]
        dist = tfd.Normal(loc=dgp.loc(x), scale=dgp.scale(x))
        ref_log_prob = dist.log_prob(data["y"])

        assert np.allclose(ref_log_prob, data["log_prob"], atol=1e-3)
        # (
        #     ggplot()
        #     + geom_point(aes(data["log_prob"], ref_log_prob))
        #     + geom_abline(slope=1, intercept=0, color="red")
        # )

    def test_deriv(self):
        key = jax.random.PRNGKey(1848)
        _, k2, _, k4 = jax.random.split(key, 4)
        shape = np.zeros(10)

        dgp = dg.PTMLocScaleDataGen(shape)

        h_deriv = jax.jacobian(dgp.transformation, argnums=0)
        data = dgp.sample(k4, 60)
        y = data["y"]

        d_auto = np.diag(h_deriv(y))
        d_man = dgp.transformation_deriv(y)

        assert np.allclose(d_auto, d_man)

    def test_deriv_nonzero(self):
        key = jax.random.PRNGKey(1848)
        k1, k2, _, k4 = jax.random.split(key, 4)
        shape = sample_shape(k1, 20).sample

        dgp = dg.PTMLocScaleDataGen(shape)

        h_deriv = jax.jacobian(dgp.transformation, argnums=0)
        data = dgp.sample(k4, 60)
        y = data["y"]

        d_auto = np.diag(h_deriv(y))
        d_man = dgp.transformation_deriv(y)

        assert np.allclose(d_auto, d_man)

    def test_nonnormal(self) -> None:
        key = jax.random.PRNGKey(1848)
        k1, k2, _, k4 = jax.random.split(key, 4)
        shape = sample_shape(k1, 20).sample

        b = jax.random.normal(k2, shape=(3,))

        def _locfn(x):
            return x @ b

        dgp = dg.PTMLocScaleDataGen(shape, loc_fn=_locfn, ncov=3)

        data = dgp.sample(k4, 60)
        x = data["x"]
        dist = tfd.Normal(loc=dgp.loc(x), scale=dgp.scale(x))
        ref_log_prob = dist.log_prob(data["y"])

        assert not np.allclose(ref_log_prob, data["log_prob"])

    def test_to_df_no_covariate(self) -> None:
        key = jax.random.PRNGKey(1848)
        k1, _, _, k4 = jax.random.split(key, 4)
        shape = sample_shape(k1, 20).sample

        dgp = dg.PTMLocScaleDataGen(shape)

        data = dgp.sample(k4, 60)
        df = dgp.to_df(data)

        assert df.shape == (60, 11)

    def test_dfgrid(self) -> None:
        key = jax.random.PRNGKey(1848)
        k1, _, _, k4 = jax.random.split(key, 4)
        shape = sample_shape(k1, 20).sample

        dgp = dg.PTMLocScaleDataGen(shape)

        z = np.linspace(-3, 3, 100)

        df = dgp.dfgrid(z)

        assert df.shape == (100, 11)
