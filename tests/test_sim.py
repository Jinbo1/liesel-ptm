from collections.abc import Iterator
from pathlib import Path

import jax
import jax.numpy as jnp
import liesel.goose as gs
import pandas as pd
import pytest

import liesel_ptm as ptm
from liesel_ptm.sim import create_data as data
from liesel_ptm.sim import sim_ptm as sim

key = jax.random.PRNGKey(1)
k1, k2 = jax.random.split(key)
shape = ptm.sample_shape(k1, nshape=15, scale=0.75).sample

pytestmark = pytest.mark.skipif(True, reason="Only tested locally")


@pytest.fixture(scope="module")
def df_loc() -> Iterator[pd.DataFrame]:
    df = data.generate_data(k2, shape, nobs=100, loc_fn=data.loc_fn, scale_fn=None)
    yield df


@pytest.fixture(scope="module")
def df_scale() -> Iterator[pd.DataFrame]:
    df = data.generate_data(
        k2, shape, nobs=100, loc_fn=data.loc_fn, scale_fn=data.scale_fn
    )
    yield df


@pytest.fixture(scope="module")
def results() -> Iterator[gs.SamplingResults]:
    yield gs.SamplingResults.pkl_load(
        "tests/resources/sim_dtm_locscale/seed_3-results.pickle"
    )


def test_generate_data_loc(df_loc):
    df = df_loc
    assert df.shape[0] == 100
    assert df.isna().sum().sum() == 0

    for i in range(4):
        assert df[f"fx{i}_loc"].mean() == pytest.approx(0.0, abs=0.1)


def test_model_fit():
    df = data.generate_data(
        k2, shape, nobs=2000, loc_fn=data.loc_fn, scale_fn=data.scale_fn
    )
    model = sim.model_setup(df, scale_terms=True)
    fit = sim.model_fit(
        3,
        model,
        warmup=300,
        posterior=1000,
        cache_path=Path("tests/resources/sim_dtm_locscale"),
        optimize_start_values=True,
    )
    assert fit is not None


def test_generate_data_scale(df_scale):
    df = df_scale
    assert df.shape[0] == 100
    assert df.isna().sum().sum() == 0

    for i in range(4):
        assert df[f"fx{i}_loc"].mean() == pytest.approx(0.0, abs=0.1)
        assert df[f"fx{i}_scale"].mean() == pytest.approx(0.0, abs=0.1)
        assert df[f"fx{i}_scale"].std() == pytest.approx(1.0, abs=0.1)


def test_create_one_dataset(tmp_path):
    data.create_one_dataset(
        seed=1,
        nobs=100,
        path=tmp_path,
        shape_scale=0.75,
        loc_fn=data.loc_fn,
        scale_fn=data.scale_fn,
        ncov=4,
    )

    assert (tmp_path / "train").exists()
    assert (tmp_path / "test").exists()
    assert (tmp_path / "train" / "train-seed_1.csv").exists()
    assert (tmp_path / "test" / "test-seed_1.csv").exists()


def test_generate_data_with_quantiles():
    df = data.generate_data(
        k2,
        shape,
        nobs=1000,
        loc_fn=data.loc_fn,
        scale_fn=data.scale_fn,
        quantiles=jnp.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]),
    )

    n_deviations = 0
    for q in [
        "0.01",
        "0.10",
        "0.20",
        "0.30",
        "0.40",
        "0.50",
        "0.60",
        "0.70",
        "0.80",
        "0.90",
        "0.99",
    ]:
        deviation = (df["y"] <= df[f"q_{q}"]).sum() - float(q) * 1000
        assert jnp.abs(deviation) < 50
        n_deviations += deviation

    assert jnp.abs(n_deviations) < 150


def test_create_study_data(tmp_path):
    data.create_study_data(
        start_seed=1,
        nobs=100,
        path=tmp_path,
        n_datasets=2,
        shape_scale=0.75,
        loc_fn=data.loc_fn,
        scale_fn=data.scale_fn,
        ncov=4,
    )

    assert (tmp_path / "train").exists()
    assert (tmp_path / "test").exists()

    assert (tmp_path / "train" / "train-seed_1.csv").exists()
    assert (tmp_path / "test" / "test-seed_1.csv").exists()

    assert (tmp_path / "train" / "train-seed_2.csv").exists()
    assert (tmp_path / "test" / "test-seed_2.csv").exists()


def test_model_setup_loc(df_loc):
    df = df_loc
    model = sim.model_setup(df)
    assert model is not None


def test_model_setup_scale(df_scale):
    df = df_scale
    model = sim.model_setup(df, scale_terms=True)
    assert model is not None


def test_log_exception_and_pass(caplog):
    def _exc():
        raise Exception("Test exception")

    with sim.log_exception_and_pass("sim"):
        _exc()

    assert "Test exception" in caplog.text


def test_model_mse(results, df_scale):
    samples = results.get_posterior_samples()
    model = sim.model_setup(df_scale, scale_terms=True)

    mse = sim.model_mse(samples, model=model, test_df=df_scale, scale_terms=True)
    # from plotnine import ggplot, aes, geom_line, facet_wrap

    # mse = mse[mse["predictor"] == "loc"]
    # (
    #     ggplot(data=mse)
    #     + geom_line(aes("x_value", "fx_true", group="predictor"))
    #     + facet_wrap("~ x_label", scales="free")
    #     + geom_line(aes("x_value", "fx_median", color="predictor"))
    # )
    mse_summary = sim.model_mse_summary(mse)

    assert mse.isna().sum().sum() == 0
    assert mse_summary.isna().sum().sum() == 0


def test_model_quantiles(results, df_scale):
    samples = results.get_posterior_samples()
    model = sim.model_setup(df_scale, scale_terms=True)

    quantiles = sim.model_quantiles(
        samples, model=model, test_df=df_scale, scale_terms=True
    )

    assert quantiles.isna().sum().sum() == 0


def test_model_dic(results, df_scale):
    samples = results.get_posterior_samples()
    model = sim.model_setup(df_scale, scale_terms=True)

    dic = sim.model_deviance(samples, model=model, df=df_scale, scale_terms=True)

    assert dic is not None


def test_model_decomposed_crps(results, df_scale):
    samples = results.get_posterior_samples()
    model = sim.model_setup(df_scale, scale_terms=True)

    df = sim.model_quantile_score(samples, model=model, df=df_scale, scale_terms=True)

    assert df is not None


def test_model_crps(results, df_scale):
    samples = results.get_posterior_samples()
    model = sim.model_setup(df_scale, scale_terms=True)

    df = sim.model_crps(samples, model=model, df=df_scale, scale_terms=True)

    assert df is not None


def test_model_analysis(results, df_scale):
    model = sim.model_setup(df_scale, scale_terms=True)
    data = sim.model_analysis(
        model,
        results,
        df_scale,
        df_scale.iloc[0:20, :],
        df_scale.iloc[0:, :],
        seed=1,
        scale_terms=True,
    )

    assert data["summary_by_quantiles"].isna().sum().sum() == 0
