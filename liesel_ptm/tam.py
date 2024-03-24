from __future__ import annotations

import logging
from itertools import chain

import arviz as az
import jax.numpy as jnp
import liesel.model as lsl
import numpy as np
import scipy
import tensorflow_probability.substrates.jax.distributions as tfd

from .bsplines import kn
from .custom_types import Array
from .inverse_fn import initial_inverse_guess, invert_fn
from .nodes import (
    BSplineBasis,
    ExpParam,
    Predictor,
    ScaledDot,
    ShapeParam,
    TransformationDist,
    addition,
)
from .nodes import cumsum_leading_zero as _cumsum_leading_zero
from .nodes import sfn as _sfn

logger = logging.getLogger(__name__)


def _transformation_scale(exp_shape: Array, dknots: float) -> Array:
    return dknots / _sfn(exp_shape)


class Normalization:
    def __init__(
        self, knots: Array, y: Array, shape_scale: ExpParam, name: str
    ) -> None:
        self.basis = BSplineBasis(
            knots, y, order=3, centered=True, name=f"{name}_basis"
        )
        self.nshape = self.basis.nparam - 1
        self.shape_scale = shape_scale

        self.dknots = np.diff(knots).mean()
        self.shape = ShapeParam(
            nparam=self.nshape, scale=self.shape_scale, name=f"{name}_shape"
        )
        exp_shape = lsl.Calc(jnp.exp, self.shape).update()

        slope_coefs = lsl.Calc(_cumsum_leading_zero, exp_shape).update()
        transformation_scale = lsl.Calc(
            _transformation_scale, exp_shape, self.dknots
        ).update()
        smooth_unshifted = ScaledDot(
            x=self.basis, coef=slope_coefs, scale=transformation_scale
        )

        self.ymean = y.mean()
        self.smooth = lsl.Calc(
            addition, smooth_unshifted, self.ymean, _name=name
        ).update()

        self.basis_deriv = lsl.Data(self.basis.deriv(), _name=f"{name}_basis_deriv")
        self.smooth_deriv = ScaledDot(
            x=self.basis_deriv, coef=slope_coefs, scale=transformation_scale
        )

        self.nuts_params = [self.shape_scale.log_var.name, self.shape.transformed_name]

    @classmethod
    def auto(
        cls, y: Array, nshape: int, shape_scale: ExpParam, name: str
    ) -> Normalization:
        knots = kn(y, order=3, n_params=nshape + 1)
        return cls(knots, y, shape_scale, name)

    def _exp_shape(self, samples: dict[str, Array]) -> Array:
        return np.exp(self.shape.shape_calc.predict(samples))

    def _slope_coef(self, samples: dict[str, Array]) -> Array:
        return _cumsum_leading_zero(self._exp_shape(samples))

    def _transf_scale(self, samples: dict[str, Array]) -> Array:
        """
        This is the correction factor to bring the function to an average slope of one.
        It is not the scale of the shape parameters.
        """
        return _transformation_scale(self._exp_shape(samples), self.dknots)

    def predict_shape(self, samples: dict[str, Array]) -> Array:
        return self.shape.shape_calc.predict(samples)

    def predict(self, samples: dict[str, Array], y: Array | None = None):
        slope_coef_samples = self._slope_coef(samples)
        transformation_scale_samples = self._transf_scale(samples)
        basis = self.basis.evaluate(y)

        unscaled_smooth = np.tensordot(basis, slope_coef_samples, axes=([1], [-1]))
        unscaled_smooth = np.moveaxis(unscaled_smooth, 0, -1)
        smooth = transformation_scale_samples * unscaled_smooth + self.ymean
        return smooth

    def predict_deriv(self, samples: dict[str, Array], y: Array | None = None):
        slope_coef_samples = self._slope_coef(samples)
        transformation_scale_samples = self._transf_scale(samples)
        basis_deriv = self.basis.deriv(y)

        unscaled_smooth = np.tensordot(
            basis_deriv, slope_coef_samples, axes=([1], [-1])
        )
        unscaled_smooth = np.moveaxis(unscaled_smooth, 0, -1)
        smooth_deriv = transformation_scale_samples * unscaled_smooth
        return smooth_deriv


def _standardization(x: Array, loc: Array, scale: Array) -> Array:
    return (x - loc) / scale


def _divide(a, b):
    return a / b


def normalization_fn(basis: BSplineBasis, y: Array, shape: Array):
    assert basis.centered

    diff = np.diff(basis.knots)
    if not jnp.allclose(diff, diff[0], 1e-3, 1e-3):
        raise ValueError("Knots must be equidistant.")
    dknots = diff.mean()

    exp_shape = np.exp(shape)
    slope_coef = _cumsum_leading_zero(exp_shape)
    transf_scale = _transformation_scale(exp_shape, dknots)

    try:
        ybasis = basis.evaluate(y)
    except ValueError as e:
        if "Data values are not in the range" not in str(e):
            raise e
        ybasis = np.full((np.shape(y)[0], basis.nparam), np.nan)
        ybasis[np.nonzero(y < basis.min)] = np.full((1, basis.nparam), -100.0)
        ybasis[np.nonzero(y > basis.max)] = np.full((1, basis.nparam), 100.0)
        inrange = np.nonzero((basis.min <= y) & (y < basis.max))
        if inrange[0]:
            ybasis[inrange] = basis.evaluate(y[inrange])

    norm_t = np.tensordot(ybasis, slope_coef, axes=([1], [-1]))
    norm = np.moveaxis(transf_scale * norm_t, 0, -1) + np.mean(basis.observed_value)
    return norm


def normalization_fn_deriv(basis: BSplineBasis, y: Array, shape: Array):
    assert basis.centered

    diff = np.diff(basis.knots)
    if not jnp.allclose(diff, diff[0], 1e-3, 1e-3):
        raise ValueError("Knots must be equidistant.")
    dknots = diff.mean()

    exp_shape = np.exp(shape)
    slope_coef = _cumsum_leading_zero(exp_shape)
    transf_scale = _transformation_scale(exp_shape, dknots)

    try:
        ybasis = basis.deriv(y)
    except ValueError as e:
        if "Data values are not in the range" not in str(e):
            raise e
        ybasis = np.full((np.shape(y)[0], basis.nparam), np.nan)
        ybasis[np.nonzero(y < basis.min)] = np.full((1, basis.nparam), 1.0)
        ybasis[np.nonzero(y > basis.max)] = np.full((1, basis.nparam), 1.0)
        inrange = np.nonzero((basis.min <= y) & (y < basis.max))
        if inrange:
            ybasis[inrange] = basis.deriv(y[inrange])

    norm_deriv_t = np.tensordot(ybasis, slope_coef, axes=([1], [-1]))
    norm_deriv = np.moveaxis(transf_scale * norm_deriv_t, 0, -1)
    return norm_deriv


def transformation_fn(
    basis: BSplineBasis, y: Array, loc: Array, scale: Array, shape: Array
) -> Array:
    for name, param in (("loc", loc), ("scale", scale), ("shape", shape)):
        if len(np.shape(param)) > 1:
            raise ValueError(f"Too many dimensions in parameter '{name}'.")
    norm = normalization_fn(basis, y, shape)
    norm = np.expand_dims(norm, -1)
    z = (norm - loc) / scale
    return np.squeeze(z).astype(np.float32)


def transformation_fn_deriv(
    basis: BSplineBasis, y: Array, scale: Array, shape: Array
) -> Array:
    for name, param in (("scale", scale), ("shape", shape)):
        if len(np.shape(param)) > 1:
            raise ValueError(f"Too many dimensions in parameter '{name}'.")
    norm_deriv = normalization_fn_deriv(basis, y, shape)
    norm_deriv = np.expand_dims(norm_deriv, -1)
    transformation_deriv = norm_deriv / scale
    return transformation_deriv


def transformation_fn_jac(
    basis: BSplineBasis, y: Array, scale: Array, shape: Array
) -> Array:
    deriv = transformation_fn_deriv(basis, y, scale, shape)
    return np.diag(np.atleast_1d(np.squeeze(deriv)))


def inverse_transformation_fn(
    z: Array,
    basis: BSplineBasis,
    loc: Array,
    scale: Array,
    shape: Array,
    initial_guess: Array | None = None,
    tol: float = 1e-4,
    maxiter: int = 200,
) -> scipy.optimize.OptimizeResult:
    def h(y):
        return transformation_fn(basis, y, loc, scale, shape)

    def h_jac(y):
        return transformation_fn_jac(basis, y, scale, shape)

    inverse_h = invert_fn(h, jac=h_jac, tol=tol, maxiter=maxiter)

    if initial_guess is None:
        z0 = h(basis.observed_value)
        initial_guess = initial_inverse_guess(basis.observed_value, z0, z)

    return inverse_h(z, initial_guess)


class TAMLocScale:
    def __init__(self, knots: Array, y: Array, shape_scale_prior: lsl.Dist) -> None:
        self.loc = Predictor("loc").update()
        self.log_scale = Predictor("log_scale").update()
        self.scale = lsl.Calc(jnp.exp, self.log_scale).update()

        self.shape_scale = ExpParam(
            10.0,
            distribution=shape_scale_prior,
            name="normalization_shape_scale",
        )
        self.normalization = Normalization(
            knots, y, self.shape_scale, name="normalization"
        )

        self.z = lsl.Calc(
            _standardization, self.normalization.smooth, self.loc, self.scale, _name="z"
        ).update()
        self.z_deriv = lsl.Calc(
            _divide, self.normalization.smooth_deriv, self.scale, _name="z_deriv"
        ).update()

        self.refdist = tfd.Normal(loc=0.0, scale=1.0)

        response_dist = TransformationDist(self.z, self.z_deriv, refdist=self.refdist)
        self.response = lsl.obs(y, response_dist, name="response").update()

    @classmethod
    def from_nparam(cls, y, nparam: int, shape_scale_prior: lsl.Dist) -> TAMLocScale:
        knots = kn(y, order=3, n_params=nparam + 1)
        return cls(knots, y, shape_scale_prior=shape_scale_prior)

    def predict_z(self, samples: dict[str, Array], y: Array, **kwargs: Array) -> Array:
        term_names = list(self.loc.terms) + list(self.log_scale.terms)
        for name in kwargs:
            if name not in term_names:
                raise ValueError(f"The name does not belong to any model term: {name}.")

        normalized = self.normalization.predict(samples, y)

        loc = self.loc.predict(samples, **kwargs)
        scale = np.exp(self.log_scale.predict(samples, **kwargs))

        loc = np.atleast_3d(loc)
        scale = np.atleast_3d(scale)
        z = (normalized - loc) / scale
        return z

    def predict_z_deriv(
        self, samples: dict[str, Array], y: Array, **kwargs: Array
    ) -> Array:
        normalized_deriv = self.normalization.predict_deriv(samples, y)
        scale = np.exp(self.log_scale.predict(samples, **kwargs))

        z_deriv = normalized_deriv / scale
        if np.any(z_deriv <= 0):
            smallest_derivative = np.min(z_deriv)
            n_seq_zero = np.sum(z_deriv <= 0)
            logger.warning(
                f"{n_seq_zero} derivative values <= 0 encountered. Smallest derivative:"
                f" {smallest_derivative}. This may be just a numerical artifact due to"
                " floating point imprecision, but it should be checked."
            )
        return z_deriv

    def predict_log_prob(
        self, samples: dict[str, Array], y: Array, **kwargs: Array
    ) -> Array:
        z = self.predict_z(samples, y, **kwargs)
        z_deriv = self.predict_z_deriv(samples, y, **kwargs)
        z_deriv = jnp.maximum(z_deriv, 1e-30)
        base_log_prob = self.refdist.log_prob(z)
        log_prob_adjustment = jnp.log(z_deriv)
        log_prob = jnp.add(base_log_prob, log_prob_adjustment)
        return log_prob

    def predict_cdf(
        self, samples: dict[str, Array], y: Array, **kwargs: Array
    ) -> Array:
        z = self.predict_z(samples, y, **kwargs)
        return self.refdist.cdf(z)

    def predict_pdf(
        self, samples: dict[str, Array], y: Array, **kwargs: Array
    ) -> Array:
        return np.exp(self.predict_log_prob(samples, y, **kwargs))

    def predict_quantile(self, samples: dict[str, Array], q: Array, **kwargs: Array):
        """
        Currently, samples must be of shape (chains, iterations, param_dimension).
        The last dimension can be omitted for scalar parameters. Basically, this
        function assumes that you enter the samples just like you got them from
        goose.
        """
        z = self.refdist.quantile(q)
        loc = self.loc.predict(samples, **kwargs)
        scale = jnp.exp(self.log_scale.predict(samples, **kwargs))
        shape = self.normalization.predict_shape(samples)
        basis = self.normalization.basis

        def ih(loc, scale, shape):
            return inverse_transformation_fn(z, basis, loc, scale, shape)

        assert loc.shape == scale.shape
        np.broadcast_shapes(loc.shape, shape.shape)

        if loc.ndim == 1:
            return ih(loc, scale, shape).x

        yhat_shape = np.broadcast_shapes(loc.shape, z.shape)
        yhat = np.empty(yhat_shape)

        if loc.ndim == 2:
            for i in range(loc.shape[0]):
                myloc = loc[i, :]
                myscale = scale[i, :]
                myshape = shape[i, :]
                yhat[i, :] = ih(myloc, myscale, myshape).x

        elif loc.ndim == 3:
            for i in range(loc.shape[0]):
                for j in range(loc.shape[1]):
                    myloc = loc[i, j, :]
                    myscale = scale[i, j, :]
                    myshape = shape[i, j, :]
                    yhat[i, j, :] = ih(myloc, myscale, myshape).x

        return yhat

    def waic(self, samples: dict[str, Array]):
        loc_terms = {term.name: None for term in self.loc.terms.values()}
        scale_terms = {term.name: None for term in self.log_scale.terms.values()}
        terms = loc_terms | scale_terms
        log_prob = self.predict_log_prob(samples, y=self.response.value, **terms)
        idata = az.convert_to_inference_data({"y": log_prob}, group="log_likelihood")

        return az.waic(idata)

    @property
    def nuts_params(self) -> list[str]:
        terms = (self.normalization, self.loc, self.log_scale)
        param_generator = (term.nuts_params for term in terms)
        return list(chain.from_iterable(param_generator))
