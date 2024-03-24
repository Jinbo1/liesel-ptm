"""
Runs a model and saves the results
"""

from pathlib import Path

import liesel.goose as gs
import liesel.model as lsl
import numpy as np
import pandas as pd
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel_ptm as ptm
import liesel_ptm.nodes as nd
import liesel_ptm.ptm_ls as lstm

fp = Path("tests/resources/normal_nonlinear_star")


def run():
    np.random.seed(2407)
    x = np.random.uniform(low=-1.0, high=1.0, size=300)
    sigma = np.exp(-0.3 + 0.2 * x)
    mu = 1.2 + 0.4 * x
    y = mu + sigma * np.random.normal(size=300)

    df = pd.DataFrame({"x": x, "y": y})
    df.to_csv(fp / "data.csv", index=False)

    def init_model(knots_lo: float = -2.0, knots_hi: float = 2.0):
        model = lstm.PTMLocScale.from_nparam(
            y, nparam=30, knots_lo=knots_lo, knots_hi=knots_hi
        )

        tau2 = nd.VarHalfCauchy(100.0, scale=25.0, name="tau2")
        model.loc_model += nd.S.pspline(x, nparam=10, tau2=tau2, name="x_loc")

        model.log_scale_model += ptm.LinearTerm(x, name="x_scale_lin")

        scale_prior = lsl.Dist(tfd.HalfCauchy, loc=0.0, scale=25.0)
        scale = ptm.ExpParam(10.0, scale_prior)
        model.loc_model += ptm.NonlinearPSpline.from_nparam(
            x, nparam=20, scale=scale, name="x_loc_nonlin"
        )

        scale_prior = lsl.Dist(tfd.HalfCauchy, loc=0.0, scale=25.0)
        scale = ptm.ExpParam(10.0, scale_prior)
        model.log_scale_model += ptm.NonlinearPSpline.from_nparam(
            x, nparam=20, scale=scale, name="x_scale_nonlin"
        )

        return model

    model = init_model()
    graph = model.build_graph(maxiter=10)

    eb = gs.EngineBuilder(seed=192023, num_chains=4)
    eb = model.setup_engine_builder(eb, graph)
    eb.set_duration(warmup_duration=500, posterior_duration=50)
    engine = eb.build()

    engine.sample_all_epochs()
    results = engine.get_results()

    results.pkl_save(fp / "results.pickle")
    model._pkl_knots(fp / "knots.pickle")


if __name__ == "__main__":
    run()
