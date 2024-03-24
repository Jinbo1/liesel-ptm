from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import tensorflow_probability.substrates.jax.distributions as tfd
from scipy import stats

from . import ptm_ls as lstm4
from .bsplines import kn
from .custom_types import Array, KeyArray
from .nodes import BSplineBasis, cholesky_ltinv, diffpen, sumzero_coef
from .tam import (
    inverse_transformation_fn,
    normalization_fn,
    transformation_fn,
    transformation_fn_deriv,
)


@dataclass
class ShapeParamSample:
    sample: Array
    latent_sample: Array
    Z: Array
    Ltinv: Array
    scale: float


def sample_shape(
    prng_key: KeyArray, nshape: int, scale: float = 1.0
) -> ShapeParamSample:
    """
    Draws a random sample of the shape parameters :math:`\\boldsymbol{\\delta}` from
    a first order random walk prior.

    Parameters
    ----------
    prng_key
        A ``jax.random.PRNGKey`` for reproducibility.
    nshape
        Number of shape parameters.
    scale
        Scale of the random walk in the shape parameter's prior.

    """
    pen = diffpen(nshape, diff=1)
    Z = sumzero_coef(nshape)
    Ltinv = cholesky_ltinv(Z.T @ pen @ Z)

    shape_z = (
        tfd.Normal(loc=0.0, scale=1.0)
        .sample(Ltinv.shape[0], seed=prng_key)
        .astype(jnp.float32)
    )

    shape = scale * (Z @ Ltinv @ shape_z)
    return ShapeParamSample(shape, shape_z, Z, Ltinv, scale)


class DataGenerator:
    """Fundamental functionality for data-generating process."""

    ncov = 1

    @staticmethod
    def cov1_linear(x: Array, b: float = 1.0) -> Array:
        """Linear function."""
        return b * x

    @staticmethod
    def cov2_ushaped(x: Array) -> Array:
        """
        Roughly u-shaped function with an overall increasing trend.
        """
        return x + ((2 * x) ** 2) / 5.5

    @staticmethod
    def cov3_oscillating(x: Array) -> Array:
        """Oscillating function with an overall decreasing trend."""
        return -x + np.pi * np.sin(np.pi * x)

    @staticmethod
    def cov4_bell(x: Array) -> Array:
        """
        Function that is based on the normal PDF, but a bit twisted.
        """
        return 0.5 * x + 15 * stats.norm.pdf(2 * (x - 0.2)) - stats.norm.pdf(x + 0.4)

    @staticmethod
    def sample_z(key: KeyArray, nobs: int) -> Array:
        """Draws samples from standard normal distribution."""
        return jax.random.normal(key, shape=(nobs,))

    @staticmethod
    def sample_covariates(
        key: KeyArray, nobs: int, ncov: int, minval: float = -2.0, maxval: float = 2.0
    ) -> Array:
        """Draws uniform samples and standardizes them afterwards."""
        x = jax.random.uniform(key, shape=(nobs, ncov), minval=minval, maxval=maxval)
        x_centered = x - jnp.mean(x, axis=0, keepdims=True)
        x_std = x_centered / jnp.std(x, axis=0, keepdims=True)
        return jnp.squeeze(x_std)

    @staticmethod
    def array_to_dict(x: Array, names_prefix: str = "x") -> dict[str, Array]:
        """Turns a 2d-array into a dict."""

        if isinstance(x, float) or x.ndim == 1:
            return {f"{names_prefix}0": x}
        elif x.ndim == 2:
            return {f"{names_prefix}{i}": x[:, i] for i in range(x.shape[-1])}
        else:
            raise ValueError(f"x should have ndim <= 2, but it has x.ndim={x.ndim}")

    def loc(self, x: Array | None) -> Array:
        raise NotImplementedError

    def scale(self, x: Array | None) -> Array:
        raise NotImplementedError

    def transformation_inverse(self, z: Array, x: Array | None = None) -> Array:
        raise NotImplementedError

    def transformation(self, y: Array, x: Array | None = None) -> Array:
        raise NotImplementedError

    def transformation_deriv(self, y: Array, x: Array | None = None) -> Array:
        return jnp.diag(jax.jacobian(self.transformation)(y, x))

    def cdf(self, y: Array, x: Array | None = None) -> Array:
        z = self.transformation(y, x)
        return tfd.Normal(loc=0.0, scale=1.0).cdf(z)

    def log_prob(self, y: Array, x: Array | None = None) -> Array:
        z = self.transformation(y, x)
        deriv = jnp.maximum(self.transformation_deriv(y, x), 1e-30)
        return tfd.Normal(loc=0.0, scale=1.0).log_prob(z) + jnp.log(deriv)

    def pdf(self, y: Array, x: Array | None = None) -> Array:
        return jnp.exp(self.log_prob(y, x))

    def quantile(self, q: Array, x: Array | None = None) -> Array:
        """Percent point function; inverse cdf."""
        z = tfd.Normal(loc=0.0, scale=1.0).quantile(q)
        y = self.transformation_inverse(z, x)
        return y

    def sample(self, key: KeyArray, nobs: int) -> dict[str, Array]:
        k1, k2 = jax.random.split(key)
        z = self.sample_z(k1, nobs)
        x = self.sample_covariates(k2, nobs, ncov=self.ncov)
        y = self.transformation_inverse(z, x)

        data = self.array_to_dict(x)
        data["y"] = y
        data["z"] = z
        data["zt"] = self.scale(x) * z + self.loc(x)

        fx = self.array_to_dict(self.loc(x), names_prefix="fx")
        data |= fx

        return data


class ExponentialDataGen(DataGenerator):
    """Data generating process based on the exponential quantile function."""

    ncov = 1

    def loc(self, x: Array | None) -> Array:
        return self.cov4_bell(x)

    def scale(self, x: Array | None) -> Array:
        return 1.0

    def transformation_inverse(self, z: Array, x: Array | None = None) -> Array:
        scale = self.scale(x)
        loc = self.loc(x)
        zt = scale * z + loc
        normalized = (zt + 8) / 16
        return tfd.Exponential(rate=1.0).quantile(normalized)

    def transformation(self, y: Array, x: Array | None = None) -> Array:
        normalized = tfd.Exponential(rate=1.0).cdf(y)
        zt = 16 * normalized - 8

        scale = self.scale(x)
        loc = self.loc(x)
        z = (zt - loc) / scale
        return z


class ExpLogitDGPBase(DataGenerator):
    def loc(self, x: Array | None) -> Array:
        return self.cov4_bell(x)

    def scale(self, x: Array | None) -> Array:
        return 1.0

    def transformation_inverse(self, z: Array, x: Array | None = None) -> Array:
        scale = self.scale(x)
        loc = self.loc(x)
        zt = scale * z + loc
        normalized = jax.scipy.special.expit(zt)
        return tfd.Exponential(rate=1.0).quantile(normalized)

    def transformation(self, y: Array, x: Array | None = None) -> Array:
        normalized = tfd.Exponential(rate=1.0).cdf(y)
        zt = jax.scipy.special.logit(normalized)

        scale = self.scale(x)
        loc = self.loc(x)
        z = (zt - loc) / scale
        return z


class TAMLocScaleDataGen(DataGenerator):
    def __init__(
        self,
        ymin: float,
        ymax: float,
        shape: Array,
        ngrid: int = 200,
        tol: float = 1e-4,
        maxiter: int = 200,
        loc_fn: Callable[[Array], Array] | None = None,
        scale_fn: Callable[[Array], Array] | None = None,
        ncov: int = 0,
    ) -> None:
        self.ygrid = np.linspace(ymin, ymax, ngrid)
        self.nparam = np.shape(shape)[-1] + 1
        self.basis = BSplineBasis.auto(self.ygrid, nparam=self.nparam, centered=True)
        self.tol = tol
        self.maxiter = maxiter
        self.shape = shape
        self.ncov = ncov
        self._loc_fn = loc_fn
        self._scale_fn = scale_fn

    def loc(self, x: Array | None) -> Array:
        if self._loc_fn is None:
            return 0.0

        if x is None:
            raise ValueError("No covariate values specified.")

        return self._loc_fn(x)

    def scale(self, x: Array | None) -> Array:
        if self._scale_fn is None:
            return 1.0

        if x is None:
            raise ValueError("No covariate values specified.")

        latent_scale = self._scale_fn(x)
        assert np.all(latent_scale > 0.0)
        return latent_scale

    def normalization(self, y: Array) -> Array:
        return normalization_fn(self.basis, y, self.shape)

    @staticmethod
    def _element_i(x, i):
        try:
            return x[i]
        except (IndexError, TypeError):
            return x

    def transformation_inverse(self, z: Array, x: Array | None = None) -> Array:
        scale = self.scale(x)
        loc = self.loc(x)

        def _h_inv(zi, loc, scale):
            return inverse_transformation_fn(
                zi,
                self.basis,
                loc,
                scale,
                self.shape,
                tol=self.tol,
                maxiter=self.maxiter,
            )

        y = np.atleast_1d(np.empty(z.shape))

        try:
            idxs = range(x.shape[0])  # type: ignore
        except AttributeError:
            try:
                idxs = range(z.shape[0])
            except (AttributeError, IndexError):
                idxs = range(1)

        for i in idxs:
            loci = self._element_i(loc, i)
            scalei = self._element_i(scale, i)
            zi = self._element_i(z, i)
            y[i] = np.squeeze(_h_inv(zi, loci, scalei).x)

        return y

    def transformation(self, y: Array, x: Array | None = None) -> Array:
        zt = self.normalization(y)

        scale = self.scale(x)
        loc = self.loc(x)
        z = (zt - loc) / scale
        return z

    def transformation_deriv(self, y: Array, x: Array | None = None) -> Array:
        scale = self.scale(x)
        return jnp.squeeze(transformation_fn_deriv(self.basis, y, scale, self.shape))

    def sample(self, key: KeyArray, nobs: int) -> dict[str, Array]:
        k1, k2 = jax.random.split(key, 2)
        basis = self.basis
        x = self.sample_covariates(k1, nobs, ncov=self.ncov)
        loc = self.loc(x)
        scale = self.scale(x)

        zmin = transformation_fn(basis, basis.min, loc, scale, self.shape)
        zmax = transformation_fn(basis, basis.max, loc, scale, self.shape)

        dist = tfd.TruncatedNormal(
            loc=0.0,
            scale=1.0,
            low=zmin.astype(np.float32),
            high=zmax.astype(np.float32),
        )
        if self.ncov > 0:
            z = jnp.squeeze(dist.sample(1, seed=k2))
        else:
            z = jnp.squeeze(dist.sample(nobs, seed=k2))

        y = self.transformation_inverse(z, x)

        data = dict()
        data["y"] = y
        data["zt"] = scale * z + loc
        data["z"] = z
        data["log_prob"] = self.log_prob(y, x)
        data["pdf"] = np.exp(data["log_prob"])
        data["cdf"] = self.cdf(y, x)
        data["latent_loc"] = jnp.squeeze(loc)
        data["latent_scale"] = jnp.squeeze(scale)
        data["x"] = x

        return data

    @staticmethod
    def to_df(data: dict[str, Array]) -> pd.DataFrame:
        x = data.pop("x")
        if x is not None:
            data |= DataGenerator.array_to_dict(x)
        return pd.DataFrame(data)


class PTMLocScaleDataGen(DataGenerator):
    """
    Draws random samples from a location-scale transformation model.

    Parameters
    ----------
    shape
        A vector of shape parameters :math:`\\boldsymbol{\\delta}` used to define the
        transformation function.
    loc_fn
        A function taking in a 2d array of covariates, returning a 1d array of
        locations.
    scale_fn
        A function taking in a 2d array of covariates, returning a 1d array of
        scales.
    ncov
        An integer, giving the number of covariates to generate.
    zmin, zmax
        Lower and upper boundary knot locations, respectively.
    use_norm_at_zero
        If ``True``, the transformation function is shifted such that :math:`h(0)=0` \
        before inversion. Should be kept this way, this option exists mainly for \
        testing purposes.

    Examples
    --------

    Example without covariates::

    >>> import liesel_ptm as ptm
    >>> import jax

    >>> key1 = jax.random.PRNGKey(21)
    >>> key2 = jax.random.PRNGKey(42)
    >>> shape = ptm.sample_shape(key1, nshape=10, scale=0.5).sample

    >>> dg = ptm.PTMLocScaleDataGen(shape=shape)
    >>> sample = dg.sample(key2, nobs=10)
    >>> sample_df = dg.to_df(sample)
    >>> sample_df[["y", "pdf"]].head()
            y       pdf
    0 -2.283257  0.087928
    1  0.430342  0.645286
    2  0.091028  0.449210
    3  0.552843  0.651007
    4  0.443428  0.648257

    Example with one covariate, using the same shape::

    >>> dg = ptm.PTMLocScaleDataGen(shape=shape, loc_fn=lambda x: 1.5*x, ncov=1)
    >>> sample = dg.sample(key2, nobs=10)
    >>> sample_df = dg.to_df(sample)
    >>> sample_df[["y", "pdf"]].head()

    """

    def __init__(
        self,
        shape: Array,
        loc_fn: Callable[[Array], Array] | None = None,
        scale_fn: Callable[[Array], Array] | None = None,
        ncov: int = 0,
        zmin: float = -2.5,
        zmax: float = 2.5,
        use_norm_at_zero: bool = True,
    ) -> None:
        self.nparam = np.shape(shape)[-1] + 1
        self.knots: Array = kn(np.array([zmin, zmax]), order=3, n_params=self.nparam)
        """Knots of the transformation function's spline segment."""

        self.shape = shape
        """Shape parameters :math:`\\boldsymbol{\\delta}`."""

        dknots = jnp.diff(self.knots).mean()
        self.slope_coef = lstm4.normalization_coef(shape, dknots)

        self.ncov = ncov
        self._loc_fn = loc_fn
        self._scale_fn = scale_fn

        dist = tfd.Normal(loc=0.0, scale=1.0)
        alpha = 0.0001
        z = dist.quantile(np.linspace(alpha, 1.0 - alpha, 1000))
        z = z / z.std()  # stabilizes the scale to exactly 1
        normalization = lstm4.NormalizationFn(self.knots, order=3)

        if use_norm_at_zero:
            self._norm_at_zero = normalization(
                np.zeros(1), self.slope_coef, jnp.zeros(1), jnp.ones(1)
            )
        else:
            self._norm_at_zero = np.zeros(1)
        zt = normalization.inverse(z, self.slope_coef, self._norm_at_zero, jnp.ones(1))

        self._norm_inv_mean = zt.mean(keepdims=True)
        self._norm_inv_scale = zt.std(keepdims=True)

    def loc(self, x: Array | None) -> Array:
        """Location function."""
        if self._loc_fn is None and x is None:
            return jnp.full((1,), 0.0)

        if self._loc_fn is None and x is not None:
            return jnp.full((x.shape[0],), 0.0)

        if x is None:
            raise ValueError("No covariate values specified.")

        loc = self._loc_fn(x)  # type: ignore
        assert loc.shape[0] == x.shape[0]
        return loc

    def scale(self, x: Array | None) -> Array:
        """Scale function."""
        if self._scale_fn is None and x is None:
            return jnp.full((1,), 1.0)

        if self._scale_fn is None and x is not None:
            return jnp.full((x.shape[0],), 1.0)

        if x is None:
            raise ValueError("No covariate values specified.")

        latent_scale = self._scale_fn(x)  # type: ignore
        assert np.all(latent_scale > 0.0)
        assert latent_scale.shape[0] == x.shape[0]

        return latent_scale

    def normalization(self, y: Array) -> Array:
        """Normalization function :math:`h(\\varepsilon)`."""
        normalization = lstm4.NormalizationFn(self.knots, order=3)
        z = normalization(y, self.slope_coef, self._norm_at_zero, jnp.ones(1))
        return z

    @staticmethod
    def _element_i(x, i):
        try:
            return x[i]
        except (IndexError, TypeError):
            return x

    def normalization_inv(self, z: Array) -> Array:
        """Inverse of the normalization function."""
        normalization = lstm4.NormalizationFn(self.knots, order=3)
        zt = normalization.inverse(z, self.slope_coef, self._norm_at_zero, jnp.ones(1))
        return zt

    def transformation_inverse(self, z: Array, x: Array | None = None) -> Array:
        """
        Inverse of the full transformation function, including the scale and location
        parts. That is, let :math:`s(y|\\boldsymbol{x}) = (y - \\mu(\\boldsymbol{x})) /
        \\sigma(\\boldsymbol{x})` and :math:`g = h \\circ s`. This method here
        evaluates :math:`g^{-1}(z | \\boldsymbol{x})`.
        """
        z = jnp.atleast_1d(z)
        scale = self.scale(x)
        loc = self.loc(x)

        zt = (self.normalization_inv(z) - self._norm_inv_mean) / self._norm_inv_scale
        y = scale * zt + loc
        return y

    def transformation(self, y: Array, x: Array | None = None) -> Array:
        """
        Full transformation function, including the scale and location
        parts. That is, let :math:`s(y|\\boldsymbol{x}) = (y - \\mu(\\boldsymbol{x})) /
        \\sigma(\\boldsymbol{x})` and :math:`g = h \\circ s`. This method here
        evaluates :math:`g(y | \\boldsymbol{x})`.
        """
        scale = self.scale(x)
        loc = self.loc(x)
        pre_std = (y - loc) / scale

        std = self._norm_inv_scale * pre_std + self._norm_inv_mean
        z = self.normalization(std)
        return z

    def transformation_deriv(self, y: Array, x: Array | None = None) -> Array:
        """
        Derivative of the full transformation function, including the scale and location
        parts. That is, let :math:`s(y|\\boldsymbol{x}) = (y - \\mu(\\boldsymbol{x})) /
        \\sigma(\\boldsymbol{x})` and :math:`g = h \\circ s`. This method
        evaluates :math:`\\frac{\\partial g(y | \\boldsymbol{x})}{\\partial y}`.
        """
        scale = self.scale(x)
        loc = self.loc(x)
        pre_std = (y - loc) / scale
        std = self._norm_inv_scale * pre_std + self._norm_inv_mean

        normalization = lstm4.NormalizationFn(self.knots, order=3)
        deriv = (
            self._norm_inv_scale
            * normalization.deriv(std, self.slope_coef, jnp.ones(1))
            / scale
        )
        return jnp.squeeze(deriv)

    def sample(self, key: KeyArray, nobs: int) -> dict[str, Array]:
        """
        Draws random samples from the transformation model implied by :attr:`.shape`,
        :meth:`.loc`, and :meth:`.scale`.

        If location and/or scale functions were defined, samples covariate values first
        using :meth:`.sample_covariates`, then samples response values.

        Parameters
        ----------
        key
            A ``jax.random.PRNGKey`` for reproducibility.
        nobs
            Number of samples to draw

        Returns
        -------
        A dictionary, holding the samples in the key ``"y"`` and the sampled covariates
        in ``"x"``, next to more information about the samples.
        """
        k1, k2 = jax.random.split(key, 2)
        x = self.sample_covariates(k1, nobs, ncov=self.ncov)

        dist = tfd.Normal(loc=0.0, scale=1.0)
        z = jnp.squeeze(dist.sample(nobs, seed=k2))
        y = self.transformation_inverse(z, x)

        loc = self.loc(x)
        scale = self.scale(x)

        data = dict()
        data["y"] = y
        data["y_std"] = (y - loc) / scale
        data["z"] = z
        data["z_deriv"] = self.transformation_deriv(y, x)
        data["log_prob"] = self.log_prob(y, x)
        data["pdf"] = np.exp(data["log_prob"])
        data["cdf"] = self.cdf(y, x)
        data["loc"] = loc
        data["scale"] = scale
        data["x"] = x
        data["std_log_prob"] = data["log_prob"] + jnp.log(scale)
        data["std_pdf"] = np.exp(data["std_log_prob"])

        return data

    @staticmethod
    def to_df(data: dict[str, Array]) -> pd.DataFrame:
        """
        Takes a data dictionary as returned by :meth:`.sample` and turns it into
        a ``pandas.DataFrame``.
        """
        x = data.pop("x")
        if x is not None:
            data |= DataGenerator.array_to_dict(x)
        return pd.DataFrame(data)

    def dfgrid(self, z: Array, x: Array | None = None) -> pd.DataFrame:
        """
        Evaluates a number of quantities given fitting arrays of ``z`` and
        ``x``. The quantities include evaluations of the cumulative distribution
        function, the probability density, and the transformation inverse.

        Parameters
        ----------
        z
            Array of observations of :math:`h(\\varepsilon)`.
        x
            Array of covariate observations.
        """
        y = self.transformation_inverse(z, x=x)
        transformation_deriv = self.transformation_deriv(y, x=x)
        log_prob = self.log_prob(y, x=x)
        loc = self.loc(x)
        scale = self.scale(x)

        std_log_prob = log_prob + jnp.log(scale)
        std_pdf = np.exp(std_log_prob)

        data = dict(
            y=y,
            y_std=(y - loc) / scale,
            z=z,
            z_deriv=transformation_deriv,
            log_prob=log_prob,
            pdf=np.exp(log_prob),
            cdf=self.cdf(y, x=x),
            loc=jnp.squeeze(loc),
            scale=jnp.squeeze(scale),
            std_log_prob=std_log_prob,
            std_pdf=std_pdf,
        )
        if x is not None:
            data |= self.array_to_dict(x)

        return pd.DataFrame(data)


def example_data(seed: int, n: int) -> pd.DataFrame:
    """
    Quickly creates an example dataframe for demonstrating the use of a penalized
    transformation model.
    """
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    shape = sample_shape(k1, nshape=8, scale=0.5).sample

    dg = PTMLocScaleDataGen(
        shape, loc_fn=lambda x: x, scale_fn=lambda x: jnp.exp(x), ncov=1
    )

    sample = dg.sample(k2, nobs=n)
    df = dg.to_df(sample)
    return df
