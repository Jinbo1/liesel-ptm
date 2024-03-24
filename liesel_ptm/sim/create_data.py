import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import pandas as pd

import liesel_ptm as ptm
from liesel_ptm.util import standardize

KeyArray = Any
Array = Any
SamplingResults = Any

logger = logging.getLogger("sim")


def loc_fn(x: Array) -> Array:
    fx0 = ptm.PTMLocScaleDataGen.cov1_linear(x[:, 0])
    fx1 = ptm.PTMLocScaleDataGen.cov2_ushaped(x[:, 1])
    fx2 = ptm.PTMLocScaleDataGen.cov3_oscillating(x[:, 2])
    fx3 = ptm.PTMLocScaleDataGen.cov4_bell(x[:, 3])
    return fx0 + fx1 + fx2 + fx3


def add_loc_fns(data: dict[str, Array]) -> dict[str, Array]:
    x = data["x"]

    data["fx0_loc"] = ptm.PTMLocScaleDataGen.cov1_linear(x[:, 0])
    data["fx1_loc"] = ptm.PTMLocScaleDataGen.cov2_ushaped(x[:, 1])
    data["fx2_loc"] = ptm.PTMLocScaleDataGen.cov3_oscillating(x[:, 2])
    data["fx3_loc"] = ptm.PTMLocScaleDataGen.cov4_bell(x[:, 3])

    data["fx0_loc"] = standardize(data["fx0_loc"], scale=False)
    data["fx1_loc"] = standardize(data["fx1_loc"], scale=False)
    data["fx2_loc"] = standardize(data["fx2_loc"], scale=False)
    data["fx3_loc"] = standardize(data["fx3_loc"], scale=False)
    return data


def scale_fn(x: Array) -> Array:
    fx0 = ptm.PTMLocScaleDataGen.cov1_linear(x[:, 0]) / 5
    fx1 = ptm.PTMLocScaleDataGen.cov2_ushaped(x[:, 1]) / 10
    fx2 = ptm.PTMLocScaleDataGen.cov3_oscillating(x[:, 2]) / 10
    fx3 = ptm.PTMLocScaleDataGen.cov4_bell(x[:, 3]) / 10
    return jnp.exp(fx0 + fx1 + fx2 + fx3)


def add_scale_fns(data: dict[str, Array]) -> dict[str, Array]:
    x = data["x"]

    data["fx0_scale"] = ptm.PTMLocScaleDataGen.cov1_linear(x[:, 0]) / 5
    data["fx1_scale"] = ptm.PTMLocScaleDataGen.cov2_ushaped(x[:, 1]) / 10
    data["fx2_scale"] = ptm.PTMLocScaleDataGen.cov3_oscillating(x[:, 2]) / 10
    data["fx3_scale"] = ptm.PTMLocScaleDataGen.cov4_bell(x[:, 3]) / 10

    data["fx0_scale"] = standardize(data["fx0_scale"])
    data["fx1_scale"] = standardize(data["fx1_scale"])
    data["fx2_scale"] = standardize(data["fx2_scale"])
    data["fx3_scale"] = standardize(data["fx3_scale"])

    return data


def generate_data(
    key: KeyArray,
    shape: Array,
    nobs: int,
    loc_fn: Callable[[Array], Array],
    scale_fn: Callable[[Array], Array] | None,
    ncov: int = 4,
    quantiles: Array | None = None,
) -> pd.DataFrame:
    dg = ptm.PTMLocScaleDataGen(
        shape,
        loc_fn=loc_fn,
        scale_fn=scale_fn,
        zmin=-3.0,
        zmax=3.0,
        ncov=ncov,
    )
    data = dg.sample(key, nobs=nobs)

    data = add_loc_fns(data)

    if scale_fn is not None:
        data = add_scale_fns(data)

    if quantiles is not None:
        for q in quantiles:
            data[f"q_{q:.2f}"] = dg.quantile(q, data["x"])

    df = dg.to_df(data)

    return df


def generate_show_data(
    shape: Array,
    values: Sequence[float],
    index: int,
    loc_fn: Callable[[Array], Array],
    scale_fn: Callable[[Array], Array] | None,
    ncov: int = 4,
):
    dg = ptm.PTMLocScaleDataGen(
        shape,
        loc_fn=loc_fn,
        scale_fn=scale_fn,
        zmin=-3.0,
        zmax=3.0,
        ncov=ncov,
    )
    z = jnp.linspace(-4.0, 4.0, 200)

    dfs = []
    for val in values:
        x = jnp.zeros((200, ncov)).at[:, index].set(val)
        show_df = dg.dfgrid(z, x=x)
        dfs.append(show_df)

    return pd.concat(dfs)


def create_one_dataset(
    seed: int,
    nobs: int,
    path: Path,
    shape_scale: float,
    loc_fn: Callable[[Array], Array],
    scale_fn: Callable[[Array], Array] | None,
    ncov: int = 4,
):
    train_path = path / "train"
    test_path = path / "test"
    show_path = path / "show"

    train_path.mkdir(exist_ok=True, parents=True)
    test_path.mkdir(exist_ok=True, parents=True)
    show_path.mkdir(exist_ok=True, parents=True)

    train_fp = train_path / f"train-seed_{seed}.csv"
    test_fp = test_path / f"test-seed_{seed}.csv"
    show_fp = show_path / f"show-seed_{seed}.csv"

    if train_fp.exists() and test_fp.exists() and show_fp.exists():
        logger.info(
            f"Dataframes {train_fp} and {test_fp} and {show_fp} "
            "already exist. Early return."
        )
        return

    shape = ptm.sample_shape(
        jax.random.PRNGKey(seed), nshape=10, scale=shape_scale
    ).sample

    train_df = generate_data(
        jax.random.PRNGKey(seed),
        shape,
        nobs,
        loc_fn=loc_fn,
        scale_fn=scale_fn,
        ncov=ncov,
    )
    quantiles = jnp.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])
    test_df = generate_data(
        jax.random.PRNGKey(seed + 1),
        shape,
        nobs=500,
        loc_fn=loc_fn,
        scale_fn=scale_fn,
        ncov=ncov,
        quantiles=quantiles,
    )

    show_df = generate_show_data(
        shape,
        values=(-1.0, 0.0, 1.0),
        index=3,
        loc_fn=loc_fn,
        scale_fn=scale_fn,
        ncov=ncov,
    )

    drop_cols = ["pdf", "loc", "scale", "std_log_prob", "std_pdf"]
    for df in (train_df, test_df, show_df):
        df.drop(columns=drop_cols, inplace=True, axis=1)

    train_df["shape_scale"] = shape_scale
    test_df["shape_scale"] = shape_scale
    show_df["shape_scale"] = shape_scale

    train_df.to_csv(train_fp, index=False)
    test_df.to_csv(test_fp, index=False)
    show_df.to_csv(show_fp, index=False)


def create_study_data(
    start_seed: int,
    nobs: int,
    path: Path,
    n_datasets: int,
    shape_scale: float,
    loc_fn: Callable[[Array], Array],
    scale_fn: Callable[[Array], Array] | None,
    ncov: int = 4,
):
    logger.info("Creating study data.")
    for seed in range(start_seed, start_seed + n_datasets):
        create_one_dataset(
            seed=seed,
            nobs=nobs,
            path=path,
            shape_scale=shape_scale,
            loc_fn=loc_fn,
            scale_fn=scale_fn,
            ncov=ncov,
        )
    logger.info("Study data created.")
