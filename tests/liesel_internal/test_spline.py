import jax.numpy as jnp
import numpy as np
from scipy.interpolate import BSpline

from liesel_ptm.liesel_internal.splines import (
    _check_data_range,
    _check_equidistant_knots,
    build_design_matrix_b_spline,
    build_design_matrix_b_spline_derivative,
    build_p_spline_matrix,
    create_equidistant_knots,
)


def test_knots_creation():
    x = jnp.arange(-1000, 1000)
    n_params = 2000
    order = 12

    knots = create_equidistant_knots(x, order, n_params)

    assert len(knots) == n_params + order + 1
    assert _check_equidistant_knots(knots)
    assert _check_data_range(x, knots, order)


def test_design_matrix_shape():
    n = 200
    x = jnp.arange(0, n)
    n_params = 40
    order = 3

    knots = create_equidistant_knots(x, order, n_params)

    X = build_design_matrix_b_spline(x, knots, order)

    assert X.shape[0] == n
    assert X.shape[1] == n_params


def test_design_matrix_vs_scipy():
    x = jnp.arange(0, 200)
    n_params = 40
    order = 3

    knots = create_equidistant_knots(x, order, n_params)

    X = build_design_matrix_b_spline(x, knots, order)
    beta = np.random.randn(X.shape[1])

    scipy_spl = BSpline(knots, beta, order)

    assert np.allclose(X @ beta, scipy_spl(x), 1e-3, 1e-3)


def test_penalty_matrix():
    d = 20

    beta = np.random.randn(d)

    r = 0

    K = build_p_spline_matrix(d, r)

    assert np.allclose(np.sum(np.diff(beta, r) ** 2), beta @ K @ beta)

    r = 1

    K = build_p_spline_matrix(d, r)

    assert np.allclose(np.sum(np.diff(beta, r) ** 2), beta @ K @ beta)

    r = 2

    K = build_p_spline_matrix(d, r)

    assert np.allclose(np.sum(np.diff(beta, r) ** 2), beta @ K @ beta)

    r = 3

    K = build_p_spline_matrix(d, r)

    assert np.allclose(np.sum(np.diff(beta, r) ** 2), beta @ K @ beta)

    r = 4

    K = build_p_spline_matrix(d, r)

    assert np.allclose(np.sum(np.diff(beta, r) ** 2), beta @ K @ beta)


def test_design_matrix_derivative_vs_scipy():
    x = jnp.arange(0, 200, dtype=np.float32)
    n_params = 40
    order = 3

    knots = create_equidistant_knots(x, order, n_params)

    X = build_design_matrix_b_spline_derivative(x, knots, order)
    beta = np.random.randn(X.shape[1])

    scipy_spl = BSpline(knots, beta, order).derivative()

    assert np.allclose(X @ beta, scipy_spl(x), 1e-3, 1e-3)
