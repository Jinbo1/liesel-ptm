import arviz as az
import jax
import liesel.goose as gs
import liesel.model as lsl
import numpy as np
import pytest
import scipy
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel_ptm as ptm
from liesel_ptm import tam as llstm
from liesel_ptm.datagen import sample_shape
from liesel_ptm.liesel_internal import splines

kn = splines.create_equidistant_knots

nparam = 7
x = np.random.uniform(size=20)

X = lsl.obs(np.random.uniform(-2, 2, size=(10, 3)), name="x1")
coef = lsl.param(np.random.uniform(0, 1, size=(3,)), name="b")


@pytest.fixture
def results():
    yield gs.engine.SamplingResults.pkl_load(
        "tests/resources/normal_linear/results.pickle"
    )


@pytest.fixture
def samples(results):
    yield results.get_posterior_samples()


@pytest.fixture
def shape_scale():
    prior = lsl.Dist(tfd.Exponential, rate=1.0)
    scale = ptm.ExpParam(1.0, prior, name="shape_scale").update()
    yield scale


class TestShapeParam:
    def test_init(self, shape_scale) -> None:
        shape = ptm.ShapeParam(8, scale=shape_scale, name="shape")
        assert shape.transformed_name == "shape_transformed"


class TestNormalization:
    def test_init(self, shape_scale) -> None:
        y = np.random.normal(size=13)
        norm = ptm.Normalization.auto(y, nshape=8, shape_scale=shape_scale, name="norm")
        assert norm.smooth is not None
        assert norm.nuts_params == ["shape_scale_transformed", "norm_shape_transformed"]

    # uncomment to plot
    # def test_plot_vars(self, shape_scale) -> None:
    #     y = np.random.normal(size=13)
    #     norm = ps.Normalization.auto(y, nshape=8, shape_scale=shape_scale, name="norm")
    #     gb = lsl.GraphBuilder().add(norm.smooth)
    #     gb.plot_vars()

    def test_predict(self, shape_scale) -> None:
        y = np.random.normal(size=50)
        norm = ptm.Normalization.auto(y, nshape=8, shape_scale=shape_scale, name="norm")

        samples = dict()
        samples[norm.shape.transformed_name] = np.random.uniform(
            -1, 1, size=(4, 7, norm.nshape - 1)
        )
        samples[norm.shape_scale.log_var.name] = np.random.uniform(0, 1, size=(4, 7, 1))

        ygrid = np.linspace(-0.5, 0.5, 14)

        prediction = norm.predict(samples, ygrid)
        assert prediction.shape == (4, 7, 14)
        assert np.all(np.diff(prediction) > 0)

    def test_predict_deriv(self, shape_scale) -> None:
        y = np.random.normal(size=50)
        norm = ptm.Normalization.auto(y, nshape=8, shape_scale=shape_scale, name="norm")

        samples = dict()
        samples[norm.shape.transformed_name] = np.random.uniform(
            -1, 1, size=(4, 7, norm.nshape - 1)
        )
        samples[norm.shape_scale.log_var.name] = np.random.uniform(0, 1, size=(4, 7, 1))

        ygrid = np.linspace(-0.5, 0.5, 14)

        prediction = norm.predict_deriv(samples, ygrid)
        assert prediction.shape == (4, 7, 14)
        assert np.all(prediction > 0)


class TestLatentLocScaleTM:
    def test_init(self) -> None:
        shape_scale_prior = lsl.Dist(tfd.Exponential, rate=1.0)
        y = np.random.normal(size=13)

        model = ptm.TAMLocScale.from_nparam(
            y, nparam=11, shape_scale_prior=shape_scale_prior
        )

        assert model is not None

    # uncomment to plot
    # def test_plot_vars(self) -> None:
    #     shape_scale_prior = lsl.Dist(tfd.Exponential, rate=1.0)
    #     y = np.random.normal(size=13)

    #     model = ps.LatentLocScaleTM.auto(
    #         y, nshape=11, shape_scale_prior=shape_scale_prior
    #     )

    #     gb = lsl.GraphBuilder().add(model.response)
    #     gb.plot_vars()

    def test_predict_z_no_loc_no_scale(self) -> None:
        """
        In this case, the model assumes loc = 0 and scale = 1.
        """
        shape_scale_prior = lsl.Dist(tfd.Exponential, rate=1.0)
        y = np.random.normal(size=50)

        model = ptm.TAMLocScale.from_nparam(
            y, nparam=11, shape_scale_prior=shape_scale_prior
        )
        norm = model.normalization
        samples = dict()
        samples[norm.shape.transformed_name] = np.random.uniform(
            -1, 1, size=(4, 7, norm.nshape - 1)
        )
        samples[norm.shape_scale.log_var.name] = np.random.uniform(0, 1, size=(4, 7))

        ygrid = np.linspace(-0.5, 0.5, 14)

        prediction = model.predict_z(samples, ygrid)
        assert prediction.shape == (4, 7, 14)

    def test_predict_z(self) -> None:
        shape_scale_prior = lsl.Dist(tfd.Exponential, rate=1.0)
        y = np.random.normal(size=50)

        model = ptm.TAMLocScale.from_nparam(
            y, nparam=11, shape_scale_prior=shape_scale_prior
        )

        x = np.linspace(-2, 2, 50)

        # model.loc += ps.Intercept("loc0")
        model.loc += ptm.LinearTerm(ptm.model_matrix(x), name="loc_x")
        model.loc += ptm.NonlinearPSpline.from_nparam(
            x,
            nparam=7,
            scale=ptm.ExpParam(1.0, lsl.Dist(tfd.Exponential, rate=1.0)),
            name="loc_x_nl",
        )

        # model.log_scale += ps.Intercept("scale0")
        model.log_scale += ptm.LinearTerm(ptm.model_matrix(x), name="scale_x")
        model.log_scale += ptm.NonlinearPSpline.from_nparam(
            x,
            nparam=7,
            scale=ptm.ExpParam(1.0, lsl.Dist(tfd.Exponential, rate=1.0)),
            name="scale_x_nl",
        )

        samples = dict()

        # shape parameters
        # ------------------------------------------------------------------------------
        norm = model.normalization
        samples[norm.shape.transformed_name] = np.random.uniform(
            -1, 1, size=(4, 7, norm.nshape - 1)
        )
        samples[norm.shape_scale.log_var.name] = np.random.uniform(0, 1, size=(4, 7))

        # latent location parameters
        # ------------------------------------------------------------------------------
        samples[model.loc.loc_x.coef.name] = np.random.uniform(-1, 1, size=(4, 7, 2))
        samples[model.loc.loc_x_nl.coef.name] = np.random.uniform(
            -1, 1, size=(4, 7, model.loc.loc_x_nl.nparam)
        )
        samples[model.loc.loc_x_nl.scale.log_var.name] = np.random.uniform(
            0.1, 1, size=(4, 7)
        )

        # latent scale parameters
        # ------------------------------------------------------------------------------
        samples[model.log_scale.scale_x.coef.name] = np.random.uniform(
            -1, 1, size=(4, 7, 2)
        )
        samples[model.log_scale.scale_x_nl.coef.name] = np.random.uniform(
            -1, 1, size=(4, 7, model.log_scale.scale_x_nl.nparam)
        )
        samples[model.log_scale.scale_x_nl.scale.log_var.name] = np.random.uniform(
            0.1, 1, size=(4, 7)
        )

        # prediction
        # ------------------------------------------------------------------------------
        ygrid = np.linspace(-0.5, 0.5, 14)

        prediction = model.predict_z(
            samples,
            ygrid,
            loc_x=np.array([[1.0, 1.0]]),
            loc_x_nl=1.0,
            scale_x=np.array([[1.0, 1.0]]),
            scale_x_nl=1.0,
        )

        assert prediction.shape == (4, 7, 14)
        assert np.all(np.diff(prediction) > 0)

    def test_predict_z_intercept(self) -> None:
        shape_scale_prior = lsl.Dist(tfd.Exponential, rate=1.0)
        y = np.random.normal(size=50)

        model = ptm.TAMLocScale.from_nparam(
            y, nparam=11, shape_scale_prior=shape_scale_prior
        )

        x = np.linspace(-2, 2, 50)

        model.loc += ptm.Intercept("loc0")
        model.loc += ptm.LinearTerm(x, name="loc_x")
        model.loc += ptm.NonlinearPSpline.from_nparam(
            x,
            nparam=7,
            scale=ptm.ExpParam(1.0, lsl.Dist(tfd.Exponential, rate=1.0)),
            name="loc_x_nl",
        )

        model.log_scale += ptm.Intercept("scale0")
        model.log_scale += ptm.LinearTerm(x, name="scale_x")
        model.log_scale += ptm.NonlinearPSpline.from_nparam(
            x,
            nparam=7,
            scale=ptm.ExpParam(1.0, lsl.Dist(tfd.Exponential, rate=1.0)),
            name="scale_x_nl",
        )

        samples = dict()

        # shape parameters
        # ------------------------------------------------------------------------------
        norm = model.normalization
        samples[norm.shape.transformed_name] = np.random.uniform(
            -1, 1, size=(4, 7, norm.nshape - 1)
        )
        samples[norm.shape_scale.log_var.name] = np.random.uniform(0, 1, size=(4, 7))

        # latent location parameters
        # ------------------------------------------------------------------------------
        samples[model.loc.loc0.name] = np.random.uniform(-1, 1, size=(4, 7))
        samples[model.loc.loc_x.coef.name] = np.random.uniform(-1, 1, size=(4, 7))
        samples[model.loc.loc_x_nl.coef.name] = np.random.uniform(
            -1, 1, size=(4, 7, model.loc.loc_x_nl.nparam)
        )
        samples[model.loc.loc_x_nl.scale.log_var.name] = np.random.uniform(
            0.1, 1, size=(4, 7)
        )

        # latent scale parameters
        # ------------------------------------------------------------------------------
        samples[model.log_scale.scale0.name] = np.random.uniform(-1, 1, size=(4, 7))
        samples[model.log_scale.scale_x.coef.name] = np.random.uniform(
            -1, 1, size=(4, 7)
        )
        samples[model.log_scale.scale_x_nl.coef.name] = np.random.uniform(
            -1, 1, size=(4, 7, model.log_scale.scale_x_nl.nparam)
        )
        samples[model.log_scale.scale_x_nl.scale.log_var.name] = np.random.uniform(
            0.1, 1, size=(4, 7)
        )

        # prediction
        # ------------------------------------------------------------------------------
        ygrid = np.linspace(-0.5, 0.5, 14)

        prediction = model.predict_z(
            samples,
            ygrid,
            loc_x=1.0,
            loc_x_nl=1.0,
            scale_x=1.0,
            scale_x_nl=1.0,
        )

        assert prediction.shape == (4, 7, 14)
        assert np.all(np.diff(prediction) > 0)

        prediction = model.predict_z_deriv(
            samples,
            ygrid,
            loc_x=1.0,
            loc_x_nl=1.0,
            scale_x=1.0,
            scale_x_nl=1.0,
        )

        assert prediction.shape == (4, 7, 14)
        assert np.all(prediction > 0)

    def test_predict_log_prob(self) -> None:
        shape_scale_prior = lsl.Dist(tfd.Exponential, rate=1.0)
        y = np.random.normal(size=50)

        model = ptm.TAMLocScale.from_nparam(
            y, nparam=11, shape_scale_prior=shape_scale_prior
        )

        x = np.linspace(-2, 2, 50)

        model.loc += ptm.Intercept("loc0")
        model.loc += ptm.LinearTerm(x, name="loc_x")
        model.loc += ptm.NonlinearPSpline.from_nparam(
            x,
            nparam=7,
            scale=ptm.ExpParam(1.0, lsl.Dist(tfd.Exponential, rate=1.0)),
            name="loc_x_nl",
        )

        model.log_scale += ptm.Intercept("scale0")
        model.log_scale += ptm.LinearTerm(x, name="scale_x")
        model.log_scale += ptm.NonlinearPSpline.from_nparam(
            x,
            nparam=7,
            scale=ptm.ExpParam(1.0, lsl.Dist(tfd.Exponential, rate=1.0)),
            name="scale_x_nl",
        )

        samples = dict()

        # shape parameters
        # ------------------------------------------------------------------------------
        norm = model.normalization
        samples[norm.shape.transformed_name] = np.zeros((4, 7, norm.nshape - 1))
        samples[norm.shape_scale.log_var.name] = np.random.uniform(0, 1, size=(4, 7))

        # latent location parameters
        # ------------------------------------------------------------------------------
        samples[model.loc.loc0.name] = np.zeros((4, 7))
        samples[model.loc.loc_x.coef.name] = np.zeros((4, 7))
        samples[model.loc.loc_x_nl.coef.name] = np.zeros(
            (4, 7, model.loc.loc_x_nl.nparam)
        )
        samples[model.loc.loc_x_nl.scale.log_var.name] = np.random.uniform(
            0.1, 1, size=(4, 7)
        )

        # latent scale parameters
        # ------------------------------------------------------------------------------
        samples[model.log_scale.scale0.name] = np.zeros((4, 7))
        samples[model.log_scale.scale_x.coef.name] = np.zeros((4, 7))
        samples[model.log_scale.scale_x_nl.coef.name] = np.zeros(
            (4, 7, model.log_scale.scale_x_nl.nparam)
        )
        samples[model.log_scale.scale_x_nl.scale.log_var.name] = np.random.uniform(
            0.1, 1, size=(4, 7)
        )

        # prediction
        # ------------------------------------------------------------------------------
        ygrid = np.linspace(-0.5, 0.5, 14)

        z = model.predict_z(
            samples,
            ygrid,
            loc_x=1.0,
            loc_x_nl=1.0,
            scale_x=1.0,
            scale_x_nl=1.0,
        )

        for i, j in zip(range(4), range(7)):
            assert z[i, j, :] == pytest.approx(ygrid, abs=1e-6)

        z_deriv = model.predict_z_deriv(
            samples,
            ygrid,
            loc_x=1.0,
            loc_x_nl=1.0,
            scale_x=1.0,
            scale_x_nl=1.0,
        )

        assert np.allclose(z_deriv, 1.0)

        log_prob = model.predict_log_prob(
            samples,
            ygrid,
            loc_x=1.0,
            loc_x_nl=1.0,
            scale_x=1.0,
            scale_x_nl=1.0,
        )

        ref_log_prob = tfd.Normal(loc=0.0, scale=1.0).log_prob(ygrid)

        for i, j in zip(range(4), range(7)):
            assert log_prob[i, j, :] == pytest.approx(ref_log_prob, abs=1e-4)

        pdf = model.predict_pdf(
            samples,
            ygrid,
            loc_x=1.0,
            loc_x_nl=1.0,
            scale_x=1.0,
            scale_x_nl=1.0,
        )

        ref_pdf = np.exp(ref_log_prob)
        for i, j in zip(range(4), range(7)):
            assert pdf[i, j, :] == pytest.approx(ref_pdf, abs=1e-4)

    def test_model(self) -> None:
        shape_scale_prior = lsl.Dist(tfd.Exponential, rate=1.0)
        y = np.random.normal(size=50)

        model = ptm.TAMLocScale.from_nparam(
            y, nparam=11, shape_scale_prior=shape_scale_prior
        )

        x = np.linspace(-2, 2, 50)

        model.loc += ptm.Intercept("loc0")
        model.loc += ptm.LinearTerm(x, name="loc_x")

        scale = ptm.ScaleHalfCauchy(10.0, scale=25.0, name="loc_x_nl_scale")
        model.loc += ptm.NonlinearPSpline.from_nparam(
            x,
            nparam=7,
            scale=scale,
            name="loc_x_nl",
        )

        model.log_scale += ptm.Intercept("scale0")
        model.log_scale += ptm.LinearTerm(x, name="scale_x")
        model.log_scale += ptm.NonlinearPSpline.from_nparam(
            x,
            nparam=7,
            scale=ptm.ScaleHalfCauchy(10.0, scale=25.0, name="scale_x_nl_scale"),
            name="scale_x_nl",
        )

        gb = lsl.GraphBuilder().add(model.response)
        lmod = gb.build_model()
        assert lmod is not None
        # lsl.plot_vars(lmod)

    def test_waic(self) -> None:
        shape_scale_prior = lsl.Dist(tfd.Exponential, rate=1.0)
        y = np.random.normal(size=50)

        model = ptm.TAMLocScale.from_nparam(
            y, nparam=11, shape_scale_prior=shape_scale_prior
        )

        x = np.linspace(-2, 2, 50)

        model.loc += ptm.Intercept("loc0")
        model.loc += ptm.LinearTerm(x, name="loc_x")
        model.loc += ptm.NonlinearPSpline.from_nparam(
            x,
            nparam=7,
            scale=ptm.ExpParam(1.0, lsl.Dist(tfd.Exponential, rate=1.0)),
            name="loc_x_nl",
        )

        model.log_scale += ptm.Intercept("scale0")
        model.log_scale += ptm.LinearTerm(x, name="scale_x")
        model.log_scale += ptm.NonlinearPSpline.from_nparam(
            x,
            nparam=7,
            scale=ptm.ExpParam(1.0, lsl.Dist(tfd.Exponential, rate=1.0)),
            name="scale_x_nl",
        )

        samples = dict()

        # shape parameters
        # ------------------------------------------------------------------------------
        norm = model.normalization
        samples[norm.shape.transformed_name] = np.zeros((4, 7, norm.nshape - 1))
        samples[norm.shape_scale.log_var.name] = np.random.uniform(0, 1, size=(4, 7))

        # latent location parameters
        # ------------------------------------------------------------------------------
        samples[model.loc.loc0.name] = np.zeros((4, 7))
        samples[model.loc.loc_x.coef.name] = np.zeros((4, 7))
        samples[model.loc.loc_x_nl.coef.name] = np.zeros(
            (4, 7, model.loc.loc_x_nl.nparam)
        )
        samples[model.loc.loc_x_nl.scale.log_var.name] = np.random.uniform(
            0.1, 1, size=(4, 7)
        )

        # latent scale parameters
        # ------------------------------------------------------------------------------
        samples[model.log_scale.scale0.name] = np.zeros((4, 7))
        samples[model.log_scale.scale_x.coef.name] = np.zeros((4, 7))
        samples[model.log_scale.scale_x_nl.coef.name] = np.zeros(
            (4, 7, model.log_scale.scale_x_nl.nparam)
        )
        samples[model.log_scale.scale_x_nl.scale.log_var.name] = np.random.uniform(
            0.1, 1, size=(4, 7)
        )

        waic = model.waic(samples)
        assert waic is not None
        assert waic["elpd_waic"] < 0

    def test_predict_quantile(self, samples):
        np.random.seed(2208)
        y = np.random.normal(size=300)

        shape_scale_prior = lsl.Dist(tfd.HalfCauchy, loc=0.0, scale=25.0)

        model = ptm.TAMLocScale.from_nparam(y, 15, shape_scale_prior)
        model.loc += ptm.Intercept("loc0")
        model.log_scale += ptm.Intercept("scale0")

        samp = {name: s[:2, :50] for name, s in samples.items()}
        yhat = model.predict_quantile(samp, np.array([0.1, 0.2]))
        assert yhat.shape == (2, 50, 2)


class TestFunctionInverse:
    def test_initial_inverse_guess(self) -> None:
        y = np.linspace(-5, 2, 200)
        z = np.exp(y)

        ynew = llstm.initial_inverse_guess(y, z, z)
        assert y == pytest.approx(ynew)

        ynew0 = llstm.initial_inverse_guess(y, z, z[0])
        assert y[0] == pytest.approx(ynew0)

    def test_inverse(self) -> None:
        y = np.linspace(-5, 2, 200)
        z = np.exp(y)

        iexp = llstm.invert_fn(np.exp)
        yhat = iexp(z, None).x

        assert yhat == pytest.approx(y, abs=1e-3)

        znew = np.random.uniform(0.1, 1, 10)
        yhat = iexp(znew, None).x
        assert yhat == pytest.approx(np.log(znew), abs=1e-3)

    def test_transformation_broadcasting(self) -> None:
        key = jax.random.PRNGKey(3)

        shape = sample_shape(key, 10).sample

        y = np.linspace(-2, 2, 100)
        knots = kn(y, n_params=11)
        basis = ptm.BSplineBasis(knots, y, centered=True)
        loc = np.full(y.shape, 0.0)
        scale = 1.0

        z = llstm.transformation_fn(basis, y, loc, scale, shape)

        assert z.shape == (100, 100)

    def test_transformation_inverse_loc(self) -> None:
        key = jax.random.PRNGKey(3)

        shape = sample_shape(key, 10).sample

        y = np.linspace(-2, 2, 100)
        knots = kn(y, n_params=11)
        basis = ptm.BSplineBasis(knots, y, centered=True)

        for loc in [0.0, 0.5, 1.0, 2.0]:

            def h(y):
                z = llstm.transformation_fn(basis, y, loc, scale=1.0, shape=shape)
                return np.squeeze(z)

            z = h(y)
            ynew0 = llstm.initial_inverse_guess(y, z, z)
            h1 = llstm.invert_fn(h)
            ynew = h1(z, ynew0).x

            assert y == pytest.approx(ynew, abs=1e-4)

    def test_transformation_inverse_loc_broadcasting(self) -> None:
        key = jax.random.PRNGKey(3)

        shape = sample_shape(key, 10).sample

        y = np.linspace(-2, 2, 100)
        knots = kn(y, n_params=11)
        basis = ptm.BSplineBasis(knots, y, centered=True)
        loc = np.array([0.0, 0.5])

        def h(y):
            z = llstm.transformation_fn(basis, y, loc, scale=1.0, shape=shape)
            return np.squeeze(z)

        z = h(y)
        h1 = llstm.invert_fn(h)
        with pytest.raises(ValueError):
            h1(z, None)

    def test_transformation_inverse_scale(self) -> None:
        key = jax.random.PRNGKey(3)

        shape = sample_shape(key, 10).sample

        y = np.linspace(-2, 2, 100)
        knots = kn(y, n_params=11)
        basis = ptm.BSplineBasis(knots, y, centered=True)

        for scale in [0.2, 0.5, 1.0, 2.0]:

            def h(y):
                z = llstm.transformation_fn(basis, y, loc=0.0, scale=scale, shape=shape)
                return np.squeeze(z)

            z = h(y)
            ynew0 = llstm.initial_inverse_guess(y, z, z)
            h1 = llstm.invert_fn(h)
            ynew = h1(z, ynew0).x

            assert y == pytest.approx(ynew, abs=1e-4)

    def test_transformation_inverse_sample_z(self) -> None:
        key = jax.random.PRNGKey(3)
        k1, k2 = jax.random.split(key)

        shape = sample_shape(k1, 10).sample

        y = np.linspace(-2, 2, 100)
        knots = kn(y, n_params=11)
        basis = ptm.BSplineBasis(knots, y, centered=True)

        for scale in [0.2, 0.5, 1.0, 2.0]:

            def h(y):
                z = llstm.transformation_fn(basis, y, loc=0.0, scale=scale, shape=shape)
                return np.squeeze(z)

            zmin, zmax = h(basis.min), h(basis.max)
            dist = tfd.TruncatedNormal(loc=0.0, scale=1.0, low=zmin, high=zmax)

            z = h(y)
            znew = dist.sample(20, k2)
            ynew0 = llstm.initial_inverse_guess(y, z, znew)
            h1 = llstm.invert_fn(h)
            ynew = h1(znew, ynew0).x

            assert znew == pytest.approx(h(ynew), abs=1e-4)

    def test_auto_transformation_inverse_sample_z(self) -> None:
        """
        Tests the directly provided inverse transformation function
        """
        key = jax.random.PRNGKey(3)
        k1, k2 = jax.random.split(key)

        shape = sample_shape(k1, 10).sample

        y = np.linspace(-2, 2, 100)
        knots = kn(y, n_params=11)
        basis = ptm.BSplineBasis(knots, y, centered=True)

        for scale in [0.2, 0.5, 1.0, 2.0]:

            def h(y):
                z = llstm.transformation_fn(basis, y, loc=0.0, scale=scale, shape=shape)
                return np.squeeze(z)

            zmin, zmax = h(basis.min), h(basis.max)
            dist = tfd.TruncatedNormal(loc=0.0, scale=1.0, low=zmin, high=zmax)

            z = h(y)
            znew = dist.sample(20, k2)
            ynew0 = llstm.initial_inverse_guess(y, z, znew)
            ynew1 = llstm.inverse_transformation_fn(
                znew, basis, 0.0, scale, shape, initial_guess=ynew0
            )
            ynew2 = llstm.inverse_transformation_fn(znew, basis, 0.0, scale, shape)

            assert znew == pytest.approx(h(ynew1.x), abs=1e-4)
            assert znew == pytest.approx(h(ynew2.x), abs=1e-4)


def test_sample_shape() -> None:
    for seed in range(1, 20):
        key = jax.random.PRNGKey(seed)
        shape = sample_shape(key, 10).sample
        assert shape.sum() == pytest.approx(0.0, abs=1e-6)


class TestTransformationFunction:
    def test_transformation(self) -> None:
        key = jax.random.PRNGKey(1)

        shape = sample_shape(key, 9).sample

        y = np.linspace(-3, 3, 200)
        knots = kn(y, n_params=10)
        basis = ptm.BSplineBasis(knots, y, centered=True)
        loc = 0.0
        scale = 1.0

        z = llstm.transformation_fn(basis, y, loc, scale, shape)

        assert z.shape == y.shape

    def test_identity(self) -> None:
        y = np.linspace(-3, 3, 200)
        shape = np.zeros(9)
        knots = kn(y, n_params=10)
        basis = ptm.BSplineBasis(knots, y, centered=True)
        loc = 0.0
        scale = 1.0

        z = llstm.transformation_fn(basis, y, loc, scale, shape)

        assert y == pytest.approx(z, abs=1e-5)

    def test_scale(self) -> None:
        y = np.linspace(-3, 3, 200)
        shape = np.zeros(9)
        knots = kn(y, n_params=10)
        basis = ptm.BSplineBasis(knots, y, centered=True)
        loc = 0.0
        scale = 2.0

        z = llstm.transformation_fn(basis, y, loc, scale, shape)

        assert y / scale == pytest.approx(z, abs=1e-5)

    def test_shift(self) -> None:
        y = np.linspace(-3, 3, 200)
        shape = np.zeros(9)
        knots = kn(y, n_params=10)
        basis = ptm.BSplineBasis(knots, y, centered=True)
        loc = 1.0
        scale = 1.0

        z = llstm.transformation_fn(basis, y, loc, scale, shape)

        assert (y - 1) == pytest.approx(z, abs=1e-5)

    def test_normalization(self) -> None:
        key = jax.random.PRNGKey(1)

        shape = sample_shape(key, 9).sample

        y = np.linspace(-3, 3, 200)
        knots = kn(y, n_params=10)

        basis = ptm.BSplineBasis(knots, y, centered=True)
        min1 = llstm.normalization_fn(basis, y[0], shape)
        min2 = llstm.normalization_fn(basis, y, shape).min()

        assert min1 == pytest.approx(min2)

    def test_normalization_deriv(self) -> None:
        y = np.linspace(-3, 3, 200)
        knots = kn(y, n_params=10)
        basis = ptm.BSplineBasis(knots, y, centered=True)
        for i in range(100, 120):
            key = jax.random.PRNGKey(i)
            shape = sample_shape(key, 9).sample
            norm_deriv = llstm.normalization_fn_deriv(basis, y, shape)

            assert norm_deriv.mean() == pytest.approx(1.0, abs=1e-1)

    def test_transformation_deriv(self) -> None:
        y = np.linspace(-3, 3, 200)
        knots = kn(y, n_params=10)
        basis = ptm.BSplineBasis(knots, y, centered=True)
        scales = (0.7, 1.5, 3.0)
        for i, scale in zip(range(120, 140), scales):
            key = jax.random.PRNGKey(i)
            shape = sample_shape(key, 9).sample
            transformation_deriv = llstm.transformation_fn_deriv(basis, y, scale, shape)

            assert transformation_deriv.mean() == pytest.approx(1 / scale, abs=1e-1)


@pytest.mark.mcmc
def test_run_mcmc():
    np.random.seed(2208)
    y = np.random.normal(size=300)

    shape_scale_prior = lsl.Dist(tfd.HalfCauchy, loc=0.0, scale=25.0)

    model = ptm.TAMLocScale.from_nparam(y, 15, shape_scale_prior)
    model.loc += ptm.Intercept("loc0")
    model.log_scale += ptm.Intercept("scale0")

    gb = lsl.GraphBuilder().add(model.response)
    lmod = gb.build_model()

    eb = gs.EngineBuilder(seed=1743, num_chains=4)
    eb.set_model(gs.LieselInterface(lmod))
    eb.set_initial_values(lmod.state)
    eb.set_duration(warmup_duration=1000, posterior_duration=1000)

    # eb.add_kernel(gs.NUTSKernel([model.normalization.shape.transformed_name]))
    eb.add_kernel(gs.NUTSKernel(["loc0", "scale0"]))
    eb.add_kernel(gs.NUTSKernel(model.normalization.nuts_params))
    eb.positions_included += [
        "normalization_tau2_root_transformed",
        "normalization_shape",
    ]

    engine = eb.build()
    engine.sample_all_epochs()
    results = engine.get_results()
    samples = results.get_posterior_samples()

    z = model.predict_z(samples, y).mean(axis=(0, 1))
    assert z == pytest.approx(y, abs=0.3)

    shape = samples["normalization_shape"].mean(axis=(0, 1))
    assert np.allclose(shape, 0.0, atol=0.2)

    loc = samples["loc0"].mean()
    assert loc == pytest.approx(0.0, abs=0.2)

    scale = samples["scale0"].mean()
    assert scale == pytest.approx(0.0, abs=0.2)

    error_df = gs.Summary(results)._error_df()
    assert error_df.relative.max() < 0.1


@pytest.mark.mcmc
def test_run_mcmc_onechain():
    np.random.seed(2208)
    y = np.random.normal(size=300)

    shape_scale_prior = lsl.Dist(tfd.HalfCauchy, loc=0.0, scale=25.0)

    model = ptm.TAMLocScale.from_nparam(y, 15, shape_scale_prior)
    model.loc += ptm.Intercept("loc0")
    model.log_scale += ptm.Intercept("scale0")

    gb = lsl.GraphBuilder().add(model.response)
    lmod = gb.build_model()

    eb = gs.EngineBuilder(seed=1743, num_chains=1)
    eb.set_model(gs.LieselInterface(lmod))
    eb.set_initial_values(lmod.state)
    eb.set_duration(warmup_duration=1000, posterior_duration=1000)

    # eb.add_kernel(gs.NUTSKernel([model.normalization.shape.transformed_name]))
    eb.add_kernel(gs.NUTSKernel(["loc0", "scale0"]))
    eb.add_kernel(gs.NUTSKernel(model.normalization.nuts_params))
    eb.positions_included += [
        "normalization_tau2_root_transformed",
        "normalization_shape",
    ]

    engine = eb.build()
    engine.sample_all_epochs()
    results = engine.get_results()
    samples = results.get_posterior_samples()

    z = model.predict_z(samples, y).mean(axis=(0, 1))
    assert z == pytest.approx(y, abs=0.3)

    shape = samples["normalization_shape"].mean(axis=(0, 1))
    assert np.allclose(shape, 0.0, atol=0.2)

    loc = samples["loc0"].mean()
    assert loc == pytest.approx(0.0, abs=0.2)

    scale = samples["scale0"].mean()
    assert scale == pytest.approx(0.0, abs=0.2)

    error_df = gs.Summary(results)._error_df()
    assert error_df.relative.max() < 0.1


@pytest.mark.mcmc
def test_run_mcmc_psplines():
    np.random.seed(2308)
    x = np.random.uniform(-2, 2, size=300)
    z = np.random.normal(size=300)
    location = 1.0 + 0.75 * x
    scale = np.exp(-0.2 + 0.2 * x)
    y = location + scale * z

    shape_scale_prior = lsl.Dist(tfd.HalfCauchy, loc=0.0, scale=25.0)

    model = ptm.TAMLocScale.from_nparam(y, 15, shape_scale_prior)
    loc_xnonlin_scale = ptm.ExpParam(
        10.0, lsl.Dist(tfd.HalfCauchy, loc=0.0, scale=25.0)
    )
    scale_xnonlin_scale = ptm.ExpParam(
        10.0, lsl.Dist(tfd.HalfCauchy, loc=0.0, scale=25.0)
    )

    model.loc += ptm.Intercept("loc0")
    model.loc += ptm.LinearTerm(x, name="loc_x")
    model.loc += ptm.NonlinearPSpline.from_nparam(
        x, nparam=10, scale=loc_xnonlin_scale, name="loc_x_ps"
    )

    model.log_scale += ptm.Intercept("scale0")
    model.log_scale += ptm.LinearTerm(x, name="scale_x")
    model.log_scale += ptm.NonlinearPSpline.from_nparam(
        x, nparam=10, scale=scale_xnonlin_scale, name="scale_x_ps"
    )

    gb = lsl.GraphBuilder().add(model.response)
    lmod = gb.build_model()

    eb = gs.EngineBuilder(seed=1459, num_chains=4)
    eb.set_model(gs.LieselInterface(lmod))
    eb.set_initial_values(lmod.state)
    eb.set_duration(warmup_duration=1000, posterior_duration=1000)

    eb.add_kernel(gs.NUTSKernel(model.loc.nuts_params))
    eb.add_kernel(gs.NUTSKernel(model.log_scale.nuts_params))
    eb.add_kernel(gs.NUTSKernel(model.normalization.nuts_params))
    eb.positions_included += [
        model.loc.loc_x_ps.smooth.name,
        model.log_scale.scale_x_ps.smooth.name,
        "normalization_shape",
        "response_log_prob",
    ]

    engine = eb.build()
    engine.sample_all_epochs()
    results = engine.get_results()
    samples = results.get_posterior_samples()

    # test intercepts
    loc0 = samples["loc0"].mean()
    assert loc0 == pytest.approx(1.0, abs=0.2)

    scale0 = samples["scale0"].mean()
    assert scale0 == pytest.approx(-0.2, abs=0.2)

    # test linear coefficients
    loc_x_coef = samples["loc_x_coef"].mean()
    assert loc_x_coef == pytest.approx(0.75, abs=0.2)

    scale_x_coef = samples["scale_x_coef"].mean()
    assert scale_x_coef == pytest.approx(0.2, abs=0.2)

    # test normalization
    shape = samples["normalization_shape"].mean(axis=(0, 1))
    assert np.allclose(shape, 0.0, atol=0.2)

    # test nonlinear effect parts
    loc_x_ps_coef = samples[model.loc.loc_x_ps.smooth.name].mean(axis=(0, 1))
    assert np.allclose(loc_x_ps_coef, 0.0, atol=0.2)

    scale_x_ps_coef = samples[model.log_scale.scale_x_ps.smooth.name].mean(axis=(0, 1))
    assert np.allclose(scale_x_ps_coef, 0.0, atol=0.2)

    error_df = gs.Summary(results)._error_df()
    assert error_df.relative.max() < 0.1

    # test correctness of waic
    log_prob = samples["response_log_prob"]
    idata = az.convert_to_inference_data({"y": log_prob}, group="log_likelihood")
    waic_manual = az.waic(idata)

    waic_auto = model.waic(samples)

    assert waic_manual["elpd_waic"] == pytest.approx(waic_auto["elpd_waic"])


@pytest.mark.mcmc
def test_run_mcmc_exp():
    np.random.seed(2208)
    y = np.random.exponential(size=300)

    shape_scale_prior = lsl.Dist(tfd.HalfCauchy, loc=0.0, scale=25.0)

    model = ptm.TAMLocScale.from_nparam(y, 15, shape_scale_prior)
    model.loc += ptm.Intercept("loc0")
    model.log_scale += ptm.Intercept("scale0")

    gb = lsl.GraphBuilder().add(model.response)
    lmod = gb.build_model()

    eb = gs.EngineBuilder(seed=1743, num_chains=4)
    eb.set_model(gs.LieselInterface(lmod))
    eb.set_initial_values(lmod.state)
    eb.set_duration(warmup_duration=1000, posterior_duration=1000)

    # eb.add_kernel(gs.NUTSKernel([model.normalization.shape.transformed_name]))
    eb.add_kernel(gs.NUTSKernel(["loc0", "scale0"]))
    eb.add_kernel(gs.NUTSKernel(model.normalization.nuts_params))
    eb.positions_included += [
        "normalization_tau2_root_transformed",
        "normalization_shape",
    ]

    engine = eb.build()
    engine.sample_all_epochs()
    results = engine.get_results()
    samples = results.get_posterior_samples()

    z = model.predict_z(samples, y).mean(axis=(0, 1))
    assert scipy.stats.normaltest(z).pvalue > 0.05

    error_df = gs.Summary(results)._error_df()
    assert error_df.relative.max() < 0.1
