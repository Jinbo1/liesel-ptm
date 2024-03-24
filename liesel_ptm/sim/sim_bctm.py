import logging
import time
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import liesel.goose as gs
import liesel_bctm as bctm
import numpy as np
import pandas as pd
from liesel_bctm.__version__ import __version__

import liesel_ptm as ptm

from ..ptm_ls import waic
from ..sampling import optimize_parameters
from ..util import log_exception_and_pass
from .sim_ptm import save_results

KeyArray = Any
Array = Any
SamplingResults = Any

logger = logging.getLogger("sim")


def model_setup(train, test, nparam: tuple[int, int] = (10, 10)):
    logger.info("Starting model setup.")
    ytrain_min, ytest_min = train.y.min(), test.y.min()
    ytrain_max, ytest_max = train.y.max(), test.y.max()

    ylo, yhi = np.min([ytrain_min, ytest_min]), np.max([ytrain_max, ytest_max])

    x0train_min, x0test_min = train.x0.min(), test.x0.min()
    x1train_min, x1test_min = train.x1.min(), test.x1.min()
    x2train_min, x2test_min = train.x2.min(), test.x2.min()
    x3train_min, x3test_min = train.x3.min(), test.x3.min()

    x0train_max, x0test_max = train.x0.max(), test.x0.max()
    x1train_max, x1test_max = train.x1.max(), test.x1.max()
    x2train_max, x2test_max = train.x2.max(), test.x2.max()
    x3train_max, x3test_max = train.x3.max(), test.x3.max()

    x0lo, x0hi = np.min([x0train_min, x0test_min]), np.max([x0train_max, x0test_max])
    x1lo, x1hi = np.min([x1train_min, x1test_min]), np.max([x1train_max, x1test_max])
    x2lo, x2hi = np.min([x2train_min, x2test_min]), np.max([x2train_max, x2test_max])
    x3lo, x3hi = np.min([x3train_min, x3test_min]), np.max([x3train_max, x3test_max])

    ctmb = (
        bctm.CTMBuilder(train)
        .add_intercept()
        .add_trafo_teprod_full(
            "y",
            "x0",
            nparam=nparam,
            a=1.0,
            b=0.001,
            name="y_x0",
            knot_boundaries=((ylo, yhi), (x0lo, x0hi)),
        )
        .add_trafo_teprod_full(
            "y",
            "x1",
            nparam=nparam,
            a=1.0,
            b=0.001,
            name="y_x1",
            knot_boundaries=((ylo, yhi), (x1lo, x1hi)),
        )
        .add_trafo_teprod_full(
            "y",
            "x2",
            nparam=nparam,
            a=1.0,
            b=0.001,
            name="y_x2",
            knot_boundaries=((ylo, yhi), (x2lo, x2hi)),
        )
        .add_trafo_teprod_full(
            "y",
            "x3",
            nparam=nparam,
            a=1.0,
            b=0.001,
            name="y_x3",
            knot_boundaries=((ylo, yhi), (x3lo, x3hi)),
        )
        .add_response("y")
    )

    logger.info("Model setup complete.")

    return ctmb


def model_fit(
    seed: int,
    ctmb: bctm.CTMBuilder,
    warmup: int,
    posterior: int,
    cache_path: Path | None = None,
) -> SamplingResults:
    model = ctmb.build_model()

    logger.info("Pre-optimizing coefficients.")
    optim_result = optimize_parameters(
        graph=model,
        params=["intercept", "y_x0_coef", "y_x1_coef", "y_x2_coef", "y_x3_coef"],
        maxiter=16000,
    )
    logger.info(
        "Pre-optimizing coefficients finished after"
        f" {optim_result.iteration} iterations."
    )

    model.state = optim_result.model_state

    logger.info("Pre-optimizing hyper-parameters.")
    for xi in range(4):
        optim_result = optimize_parameters(
            graph=model,
            params=[f"y_x{xi}_igvar", f"y_x{xi}_penalty_weight"],
            maxiter=4000,
        )
        logger.info(
            f"Pre-optimizing hyper-parameters for x{xi} finished after"
            f" {optim_result.iteration} iterations."
        )

        model.state = optim_result.model_state

    eb = bctm.ctm_mcmc(model, seed=seed, num_chains=4)
    eb.set_duration(warmup_duration=warmup, posterior_duration=posterior)
    eb.set_initial_values(optim_result.model_state)

    mcmc_start = time.time()
    if cache_path is not None:
        logger.info(f"Caching results at '{cache_path}'.")
        results = ptm.cache_results(eb, cache_path)
    else:
        logger.info("Not caching results.")
        engine = eb.build()
        logger.info("Engine successfully built. Starting posterior sampling.")
        engine.sample_all_epochs()
        results = engine.get_results()
    mcmc_end = time.time()
    mcmc_duration = mcmc_end - mcmc_start

    logger.info("Posterior sampling finished.")
    return results, None, mcmc_duration


def kwargs_helper(y: Array, x: Array) -> dict[str, Array]:
    kwargs = {f"y_x{i}": (y, x[:, i]) for i in range(x.shape[-1])}

    return kwargs


def model_mad(
    samples: dict[str, Array],
    ctmb: bctm.CTMBuilder,
    test_df: pd.DataFrame,
):
    y = test_df["y"].to_numpy()
    x = test_df.loc[:, test_df.columns.str.startswith("x")].to_numpy()

    kwargs = kwargs_helper(y, x)
    pred = bctm.ConditionalPredictions(samples, ctmb, **kwargs)

    cdf = pred.cdf()
    true_cdf = test_df["cdf"].to_numpy()
    absdiff = jnp.abs(true_cdf - cdf).mean(axis=-1, keepdims=True)
    mad_df = ptm.summarise_by_quantiles(absdiff, names_prefix="mad_cdf")
    logger.info("Returning mean absolute difference dataframe.")
    return mad_df


def model_waic(
    samples: dict[str, Array],
    ctmb: bctm.CTMBuilder,
    df: pd.DataFrame,
):
    y = df["y"].to_numpy()
    x = df.loc[:, df.columns.str.startswith("x")].to_numpy()

    kwargs = kwargs_helper(y, x)
    pred = bctm.ConditionalPredictions(samples, ctmb, **kwargs)

    log_prob = pred.log_prob()
    logger.info("Returning WAIC dataframe.")
    return waic(log_prob_samples=log_prob)


def model_deviance(
    samples: dict[str, Array],
    ctmb: bctm.CTMBuilder,
    df: pd.DataFrame,
):
    y = df["y"].to_numpy()
    x = df.loc[:, df.columns.str.startswith("x")].to_numpy()

    kwargs = kwargs_helper(y, x)
    pred = bctm.ConditionalPredictions(samples, ctmb, **kwargs)
    log_prob = pred.log_prob().sum(axis=-1)

    deviance = -2 * log_prob
    mean_deviance = deviance.mean()

    sample_means = ptm.sample_means(samples, keepdims=True)
    pred = bctm.ConditionalPredictions(sample_means, ctmb, **kwargs)
    deviance_at_mean = jnp.squeeze(-2 * pred.log_prob().sum(axis=-1))

    dic_p = mean_deviance - deviance_at_mean
    dic = mean_deviance + dic_p
    dic_elpd = -dic / 2

    dic_df = pd.DataFrame(
        {
            "dic_elpd": dic_elpd,
            "dic_p": dic_p,
            "dic": dic,
        },
        index=[0],
    )
    logger.info("Returning DIC dataframe.")
    return dic_df


def model_kld(
    samples: dict[str, Array],
    ctmb: bctm.ConditionalPredictions,
    test_df: pd.DataFrame,
):
    y = test_df["y"].to_numpy()
    x = test_df.loc[:, test_df.columns.str.startswith("x")].to_numpy()

    kwargs = kwargs_helper(y, x)
    pred = bctm.ConditionalPredictions(samples, ctmb, **kwargs)

    log_pdf = pred.log_prob()
    true_log_pdf = test_df["log_prob"].to_numpy()
    kld = jnp.mean(true_log_pdf - log_pdf, axis=-1, keepdims=True)
    kld_df = ptm.summarise_by_quantiles(kld, names_prefix="kld")
    logger.info("Returning KLD dataframe.")
    return kld_df


def kwargs_helper_fixed(y, value: float, index: int, x) -> dict[str, Array]:
    """
    Sets *one* covariate value to the fixed value, fixing the others at zero.
    """
    x = jnp.zeros((y.shape[0], x.shape[-1]))
    x = x.at[:, index].set(value)

    return kwargs_helper(y, x)


def model_analysis(
    ctmb: bctm.CTMBuilder,
    results: SamplingResults,
    df: pd.DataFrame,
    test_df: pd.DataFrame,
    show_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    logger.info("Starting model analysis.")

    samples = results.get_posterior_samples()
    summary = gs.Summary(results)

    data = dict()

    data["param_df"] = summary.to_dataframe().reset_index()
    data["error_df"] = summary._error_df().reset_index()

    x = df.loc[:, df.columns.str.startswith("x")].to_numpy()

    y = jnp.linspace(show_df.y.min(), show_df.y.max(), 200)

    with log_exception_and_pass(logger):
        summaries = []
        index = 3

        for val in [-1.0, 0.0, 1.0]:
            kwargs = kwargs_helper_fixed(y, val, index, x)
            pred = bctm.ConditionalPredictions(samples, ctmb, **kwargs)

            pdf_samples = pred.pdf()
            summary = ptm.summarise_by_quantiles(pdf_samples, names_prefix="pdf")
            summary["y"] = y
            summary["x0"] = 0.0
            summary["x1"] = 0.0
            summary["x2"] = 0.0
            summary["x3"] = 0.0
            summary[f"x{index}"] = val

            summaries.append(summary)

        data["summary_by_quantiles"] = pd.concat(summaries)

    with log_exception_and_pass(logger):
        data["mad_df"] = model_mad(samples, ctmb, test_df)

    with log_exception_and_pass(logger):
        data["kld_df"] = model_kld(samples, ctmb, test_df)

    with log_exception_and_pass(logger):
        data["waic_df"] = model_waic(samples, ctmb, df)

    with log_exception_and_pass(logger):
        data["dic_df"] = model_deviance(samples, ctmb, df)

    with log_exception_and_pass(logger):
        data["waic_df_test"] = model_waic(samples, ctmb, test_df)

    with log_exception_and_pass(logger):
        data["dic_df_test"] = model_deviance(samples, ctmb, test_df)

    logger.info("Finished model analysis.")
    return data


def _one_run(
    seed: int,
    data_path: str | Path,
    out_path: str | Path,
    warmup: int,
    posterior: int,
    identifier: str,
    n: int,
    nparam: tuple[int, int],
    id_data: dict[str, str],
    cache_path: Path | str | None = None,
    skip_if_results_exist: bool = True,
):
    prefix = f"{identifier}-seed_{seed}"
    data_path = Path(data_path).resolve()
    out_path = Path(out_path).resolve()

    if cache_path is not None:
        cache_file = Path(cache_path) / f"{prefix}-results.pickle"
    else:
        cache_file = None

    logger.info(f"Starting run. Seed: {seed}.")

    train_path = data_path / "train" / f"train-seed_{seed}.csv"
    test_path = data_path / "test" / f"test-seed_{seed}.csv"
    show_path = data_path / "show" / f"show-seed_{seed}.csv"
    out = out_path / "out"
    out.mkdir(exist_ok=True, parents=True)

    info_df_path = out / "info" / f"{prefix}-info.csv"

    if skip_if_results_exist and info_df_path.exists():
        logger.warning(f"{info_df_path} exists. Skipping this run.")
        return

    cache_path = Path(cache_path).resolve() if cache_path is not None else None

    df = pd.read_csv(train_path)
    df = df.iloc[:n, :]
    test_df = pd.read_csv(test_path)
    show_df = pd.read_csv(show_path)

    ctmb = model_setup(df, test_df, nparam=nparam)

    fit = model_fit(
        seed,
        ctmb,
        warmup=warmup,
        posterior=posterior,
        cache_path=cache_file,
    )
    results, pre_opt_duration, mcmc_duration = fit

    info: dict[str, Any] = dict()
    info["seed"] = seed
    info["df_train"] = str(train_path)
    info["df_test"] = str(test_path)
    info["pre_opt_duration"] = pre_opt_duration
    info["mcmc_duration"] = mcmc_duration
    info["nobs"] = df.shape[0]
    info["identifier"] = identifier
    info["version"] = f"liesel_bctm-{__version__}"

    data = model_analysis(ctmb, results, df, test_df, show_df)
    data["info"] = pd.DataFrame(info, index=[0])

    additional_info = {"nparam_y": nparam[0], "nparam_x": nparam[1]}

    save_results(out, prefix, data, additional_info=additional_info | id_data)

    logger.info(f"Finished run. Seed: {seed}.")
    if cache_file is None:
        return
    logger.info(f"Deleting cache file. Seed: {seed}.")
    cache_file.unlink()
    if cache_file.exists():
        logger.error("Deleting cache file unsuccessful.")
    else:
        logger.info("Deleting cache file successful.")


def one_run(
    seed: int,
    data_path: str | Path,
    out_path: str | Path,
    warmup: int,
    posterior: int,
    identifier: str,
    n: int,
    nparam_y: int,
    nparam_x: int,
    id_data: dict[str, str],
    cache_path: Path | str | None = None,
    skip_if_results_exist: bool = True,
):
    try:
        logger = logging.getLogger("sim")
        mailog = logging.getLogger("mail")

        _one_run(
            seed,
            data_path,
            out_path,
            warmup,
            posterior,
            identifier=identifier,
            n=n,
            nparam=(nparam_y, nparam_x),
            cache_path=cache_path,
            skip_if_results_exist=skip_if_results_exist,
            id_data=id_data,
        )
    except Exception:
        logger.exception(f"Exception in run. Seed: {seed}. Identifier: {identifier}")
        mailog.exception(f"Exception in run. Seed: {seed}. Identifier: {identifier}")
