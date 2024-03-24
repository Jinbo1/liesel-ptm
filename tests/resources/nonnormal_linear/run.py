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
    x = np.random.uniform(low=-1.0, high=1.0, size=100)
    sigma = np.exp(-0.3 + 0.2 * x)
    mu = 1.2 + 0.4 * x
    y = mu + sigma * np.random.exponential(size=100)

    df = pd.DataFrame({"x": x, "y": y})
    df.to_csv("tests/resources/nonnormal_linear/data.csv", index=False)

    def init_model(knots_lo: float = -2.0, knots_hi: float = 2.0):
        model = lstm.PTMLocScale.from_nparam(
            y, nparam=30, knots_lo=knots_lo, knots_hi=knots_hi
        )
        model.loc_model += ptm.LinearTerm(x, name="x_loc")
        model.log_scale_model += ptm.LinearTerm(x, name="x_scale")
        return model

    model = init_model()
    model, position = model.optimize_knots()
    graph = model.build_graph(position=position)

    eb = gs.EngineBuilder(seed=192023, num_chains=2)
    eb.set_duration(warmup_duration=300, posterior_duration=20)
    model.setup_engine_builder(eb, graph)
    eb.positions_included += model.normalization.mean_and_sd_names
    eb.positions_included += ["normalization_coef"]
    eb.positions_included += ["unscaled_normalization_and_deriv"]
    eb.positions_included += ["scaled_normalization"]
    eb.positions_included += ["scaled_normalization_deriv"]
    eb.positions_included += ["residuals"]
    eb.positions_included += ["z_deriv"]

    engine = eb.build()
    engine.sample_all_epochs()
    results = engine.get_results()

    results.pkl_save("tests/resources/nonnormal_linear/results.pickle")
    model._pkl_knots("tests/resources/nonnormal_linear/knots.pickle")


if __name__ == "__main__":
    run()
