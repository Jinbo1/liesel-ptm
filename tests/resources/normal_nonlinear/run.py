"""
Runs a model and saves the results
"""

import liesel.goose as gs
import numpy as np
import pandas as pd

import liesel_ptm as ptm
import liesel_ptm.ptm_ls as lstm


def run():
    np.random.seed(2407)
    x = np.random.uniform(low=-1.0, high=1.0, size=300)
    sigma = np.exp(-0.3 + 0.2 * x)
    mu = 1.2 + 0.4 * x
    y = mu + sigma * np.random.normal(size=300)

    df = pd.DataFrame({"x": x, "y": y})
    df.to_csv("tests/resources/normal_nonlinear/data.csv", index=False)

    def init_model(knots_lo: float = -2.0, knots_hi: float = 2.0):
        model = lstm.PTMLocScale.from_nparam(
            y, nparam=30, knots_lo=knots_lo, knots_hi=knots_hi
        )
        model.loc_model += ptm.LinearTerm(x, name="x_loc_lin")
        model.log_scale_model += ptm.LinearTerm(x, name="x_scale_lin")

        scale = ptm.ScaleHalfCauchy(10.0, scale=25.0, name="loc_tau")
        model.loc_model += ptm.NonlinearPSpline.from_nparam(
            x, nparam=20, scale=scale, name="x_loc_nonlin"
        )

        scale = ptm.ScaleHalfCauchy(10.0, scale=25.0, name="scale_tau")
        model.log_scale_model += ptm.NonlinearPSpline.from_nparam(
            x, nparam=20, scale=scale, name="x_scale_nonlin"
        )

        return model

    model = init_model()
    model, pos = model.optimize_knots()
    graph = model.build_graph(position=pos)

    eb = gs.EngineBuilder(seed=192023, num_chains=2)
    eb.set_duration(warmup_duration=1000, posterior_duration=500)
    eb = model.setup_engine_builder(eb, graph)
    eb.positions_included += model.normalization.mean_and_sd_names

    engine = eb.build()
    engine.sample_all_epochs()
    results = engine.get_results()

    results.pkl_save("tests/resources/normal_nonlinear/results.pickle")
    model._pkl_knots("tests/resources/normal_nonlinear/knots.pickle")


if __name__ == "__main__":
    run()
