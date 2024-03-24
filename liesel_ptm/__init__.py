import logging

from .bsplines import (
    BSplineApprox,
    ExtrapBSplineApprox,
    avg_slope_bspline,
    bspline_basis,
    bspline_basis_deriv,
    bspline_basis_deriv2,
    kn,
)
from .datagen import PTMLocScaleDataGen, TAMLocScaleDataGen, example_data, sample_shape
from .nodes import (
    BasisDot,
    BSplineBasis,
    ConstantPriorScalingFactor,
    Dot,
    ExpParam,
    Intercept,
    LinearSmooth,
    LinearTerm,
    MISpline,
    NonlinearPSpline,
    Predictor,
    PSpline,
    RandomIntercept,
    RandomInterceptSumZero,
    S,
    ScaledBasisDot,
    ScaledDot,
    ScaleHalfCauchy,
    ScaleInverseGamma,
    ScaleWeibull,
    StrAT,
    StructuredAdditiveTerm,
    SymmetricallyBoundedScalar,
    TransformationDist,
    TransformedVar,
    TruncatedNormalOmega,
    VarHalfCauchy,
    VarInverseGamma,
    VarWeibull,
    bs,
    cholesky_ltinv,
    diffpen,
    model_matrix,
    normalization_coef,
    nullspace_remover,
    sumzero_coef,
    sumzero_term,
)
from .optim import OptimResult, history_to_df, optim_flat
from .ptm_ls import PTMLocScale, PTMLocScalePredictions, ShapePrior, state_to_samples
from .sampling import (
    cache_results,
    get_log_lik_fn,
    get_log_prob_fn,
    kwargs_full,
    kwargs_lin,
    kwargs_loc,
    kwargs_loc_lin,
    kwargs_scale,
    kwargs_scale_lin,
    optimize_parameters,
    sample_means,
    sample_quantiles,
    summarise_by_quantiles,
    summarise_by_samples,
)
from .tam import Normalization, ShapeParam, TAMLocScale, normalization_fn
from .var import Var


def setup_logger() -> None:
    """
    Sets up a basic ``StreamHandler`` that prints log messages to the terminal.
    The default log level of the ``StreamHandler`` is set to "info".

    The global log level for Liesel can be adjusted like this::

        import logging
        logger = logging.getLogger("liesel")
        logger.level = logging.WARNING

    This will set the log level to "warning".
    """

    # We adjust only our library's logger
    logger = logging.getLogger("liesel_ptm")

    # This is the level that will in principle be handled by the logger.
    # If it is set, for example, to logging.WARNING, this logger will never
    # emit messages of a level below warning
    logger.setLevel(logging.DEBUG)

    # By setting this to False, we prevent the Liesel log messages from being passed on
    # to the root logger. This prevents duplication of the log messages
    logger.propagate = False

    # This is the default handler that we set for our log messages
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)

    # We define the format of log messages for this handler
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)


setup_logger()
