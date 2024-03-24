"""
Runs a model and saves the results
"""

import liesel.goose as gs
import liesel.model as lsl
import numpy as np
import tensorflow_probability.substrates.jax.distributions as tfd

import liesel_ptm as ptm


def run():
    np.random.seed(2208)
    y = np.random.normal(size=300)

    shape_scale_prior = lsl.Dist(tfd.HalfCauchy, loc=0.0, scale=25.0)

    model = ptm.TAMLocScale.from_nparam(y, 15, shape_scale_prior)
    model.loc += ptm.Intercept("loc0")
    model.log_scale += ptm.Intercept("scale0")

    gb = lsl.GraphBuilder().add(model.response)
    lmod = gb.build_model()

    eb = gs.EngineBuilder(seed=1743, num_chains=4)
    eb.set_model(gs.LieselInterface(lmod))
    eb.set_initial_values(lmod.state)
    eb.set_duration(warmup_duration=1000, posterior_duration=1000)

    # eb.add_kernel(gs.NUTSKernel([model.normalization.shape.transformed_name]))
    eb.add_kernel(gs.NUTSKernel(["loc0", "scale0"]))
    eb.add_kernel(gs.NUTSKernel(model.normalization.nuts_params))
    eb.positions_included += [
        "normalization_tau2_root_transformed",
        "normalization_shape",
    ]

    engine = eb.build()
    engine.sample_all_epochs()
    results = engine.get_results()
    results.pkl_save("tests/resources/normal_linear/results.pickle")


if __name__ == "__main__":
    run()
