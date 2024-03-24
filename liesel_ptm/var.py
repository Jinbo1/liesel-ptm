from __future__ import annotations

import liesel.model as lsl
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from liesel.model.nodes import ArgGroup, Calc, InputGroup


def _transform_var_with_bijector_instance(
    var: lsl.Var, bijector_inst: tfb.Bijector
) -> lsl.Var:
    InputDist = var.dist_node.distribution
    inputs = var.dist_node.inputs
    kwinputs = var.dist_node.kwinputs

    bijector_inv = tfb.Invert(bijector_inst)

    def transform_dist(*args, **kwargs):
        return tfd.TransformedDistribution(InputDist(*args, **kwargs), bijector_inv)

    transformed_dist = lsl.Dist(
        transform_dist,
        *inputs,
        **kwinputs,
        _name=var.dist_node.name,
        _needs_seed=var.dist_node.needs_seed,
    )

    transformed_dist.per_obs = var.dist_node.per_obs

    transformed_var = lsl.Var(
        bijector_inv.forward(var.value),
        transformed_dist,
        name=f"{var.name}_transformed",
    )

    var.value_node = lsl.Calc(bijector_inst.forward, transformed_var)
    return transformed_var


def _transform_var_with_bijector_class(
    var: lsl.Var, bijector_cls: type[tfb.Bijector], *args, **kwargs
) -> lsl.Var:
    InputDist = var.dist_node.distribution

    dist_inputs = InputGroup(
        *var.dist_node.inputs,
        **var.dist_node.kwinputs,  # type: ignore
    )

    bijector_inputs = InputGroup(*args, **kwargs)

    # define distribution "class" for the transformed var
    def transform_dist(dist_args: ArgGroup, bijector_args: ArgGroup):
        tfp_dist = InputDist(*dist_args.args, **dist_args.kwargs)

        bijector_inst = bijector_cls(*bijector_args.args, **bijector_args.kwargs)
        bijector_inv = tfb.Invert(bijector_inst)

        transformed_dist = tfd.TransformedDistribution(
            tfp_dist, bijector_inv, validate_args=tfp_dist.validate_args
        )

        return transformed_dist

    dist_node_transformed = lsl.Dist(
        transform_dist,
        dist_inputs,
        bijector_inputs,
        _name=var.dist_node.name,
        _needs_seed=var.dist_node.needs_seed,
    )

    dist_node_transformed.per_obs = var.dist_node.per_obs

    bijector_obj = dist_node_transformed.init_dist().bijector

    transformed_var = lsl.Var(
        bijector_obj.forward(var.value),
        dist_node_transformed,
        name=f"{var.name}_transformed",
    )

    def bijector_fn(value, dist_inputs, bijector_inputs):
        bijector = transform_dist(dist_inputs, bijector_inputs).bijector
        return bijector.inverse(value)

    var.value_node = Calc(bijector_fn, transformed_var, dist_inputs, bijector_inputs)

    return transformed_var


class Var(lsl.Var):
    def __getattr__(self, name):
        if name.startswith("__"):  # ensures, for example, that copying works.
            raise AttributeError
        return getattr(self.value_node, name)

    def transform(
        self,
        bijector: type[tfb.Bijector] | tfb.Bijector | None,
        *bijector_args,
        **bijector_kwargs,
    ) -> lsl.Var:
        """
        Transforms the variable by adding a new transformed variable as an input.

        Creates a new variable on the transformed space the accordingly
        transformed distribution, turning the original variable into a weak variable
        without an associated distribution.

        The value of the attribute :attr:`~liesel.model.nodes.Var.parameter` is
        transferred to the transformed variable and set to ``False`` on the original
        variable. The attributes :attr:`~liesel.model.nodes.Var.observed` and
        :attr:`~liesel.model.nodes.Var.role` are set to the default values for
        the transformed variable and remain unchanged on the original variable.

        Parameters
        ----------
        bijector
            The bijector used to map the new transformed variable to this variable \
            (forward transformation). If ``None``, the experimental default event \
            space bijector (see TFP documentation) is used. If a bijector class is \
            passed, it is instantiated with the arguments ``bijector_args`` and \
            ``bijector_kwargs``. If a bijector instance is passed, it is used \
            directly.
        bijector_args
            The arguments passed on to the init function of the bijector.
        bijector_kwargs
            The keyword arguments passed on to the init function of the bijector.

        Returns
        -------
        The new transformed variable which acts as an input to this variable.

        Raises
        ------
        RuntimeError
            If the variable is weak, has no TFP distribution, the distribution has
            no default event space bijector and the argument ``bijector`` is ``None``,
            or the local model for the variable cannot be built.

        Notes
        -----
        Assumes that the distribution of this variable is a distribution from
        ``tensorflow_probability.subtrates.jax.distributions``.

        Examples
        --------

        >>> import tensorflow_probability.substrates.jax.distributions as tfd
        >>> import tensorflow_probability.substrates.jax.bijectors as tfb

        Assume we have a variable ``scale`` that is constrained to be positive, and
        we want to include the log-transformation of this variable in the model.
        We first set up the parameter var with its distribution:

        >>> prior = lsl.Dist(tfd.HalfCauchy, loc=0.0, scale=25.0)
        >>> scale = lsl.param(1.0, prior, name="scale")

        The we transform the variable to the log-scale:

        >>> log_scale = scale.transform(tfb.Exp())
        >>> log_scale
        Var(name="scale_transformed")

        Now the ``log_scale`` has a log probability, and the ``scale`` variable does
        not:

        >>> log_scale.update().log_prob
        Array(-3.6720574, dtype=float32)

        >>> scale.update().log_prob
        0.0
        """
        if self.weak:
            raise RuntimeError(f"{repr(self)} is weak")

        if self.dist_node is None:  # type: ignore
            raise RuntimeError(f"{repr(self)} has no distribution")

        if isinstance(bijector, type) and not bijector_args and not bijector_kwargs:
            raise RuntimeError(
                "You passed a bijector class instead of an instance, but did not "
                "provide any arguments. You should either provide arguments "
                "or pass an instance of the bijector class."
            )

        if not isinstance(bijector, type) and (bijector_args or bijector_kwargs):
            raise RuntimeError(
                "You passed a bijector instance and "
                "nonempty bijector arguments. You should either initialise your "
                "bijector directly with the arguments, or pass a bijector class "
                "instead."
            )

        try:
            lsl.Model([self])
        except Exception:
            raise RuntimeError(f"Cannot build local model for {self}")

        # avoid infinite recursion
        self.auto_transform = False

        if isinstance(bijector, type):
            tvar = _transform_var_with_bijector_class(
                self, bijector, *bijector_args, **bijector_kwargs
            )
        elif bijector is None:
            dist_inst = self.dist_node.init_dist()  # type: ignore
            bijector = dist_inst.experimental_default_event_space_bijector
            tvar = _transform_var_with_bijector_class(
                self, bijector, *bijector_args, **bijector_kwargs
            )
        else:
            tvar = _transform_var_with_bijector_instance(self, bijector)

        tvar.parameter = self.parameter  # type: ignore
        self.parameter = False
        self.dist_node = None

        return tvar
