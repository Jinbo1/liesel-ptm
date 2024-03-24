.. liesel-ptm documentation master file, created by
   sphinx-quickstart on Mon Jul  3 09:59:41 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Penalized Transformation Models in Liesel
==========================================


Installation
------------

The library can be installed from GitHub:

.. code:: bash

    $ pip install git+https://github.com/liesel-devs/liesel-ptm.git#egg=liesel_ptm



Acknowledgements
----------------

Liesel-PTM is developed by Johannes Brachem with support from Paul Wiemann and
Thomas Kneib at the `University of Göttingen <https://www.uni-goettingen.de/en>`_.
As a specialized extension, Liesel-PTM belongs to the Liesel project.
We are
grateful to the `German Research Foundation (DFG) <https://www.dfg.de/en>`_ for funding the development
through grant 443179956.

.. image:: https://raw.githubusercontent.com/liesel-devs/liesel/main/docs/source/_static/uni-goe.svg
   :alt: University of Göttingen

.. image:: https://raw.githubusercontent.com/liesel-devs/liesel/main/docs/source/_static/funded-by-dfg.svg
   :alt: Funded by DFG


API Reference
-------------

.. rubric:: Model

.. autosummary::
    :toctree: generated
    :caption: Model
    :recursive:
    :nosignatures:

    ~liesel_ptm.PTMLocScale
    ~liesel_ptm.PTMLocScalePredictions
    ~liesel_ptm.ShapePrior
    ~liesel_ptm.Predictor

.. rubric:: Covariate terms

.. autosummary::
    :toctree: generated
    :caption: Covariate terms
    :recursive:
    :nosignatures:

    ~liesel_ptm.LinearTerm
    ~liesel_ptm.StructuredAdditiveTerm
    ~liesel_ptm.PSpline
    ~liesel_ptm.RandomIntercept
    ~liesel_ptm.NonLinearPSpline
    ~liesel_ptm.MiSpline

.. rubric:: Variable classes

.. autosummary::
    :toctree: generated
    :caption: Variable classes
    :recursive:
    :nosignatures:

    ~liesel_ptm.VarWeibull
    ~liesel_ptm.VarInverseGamma
    ~liesel_ptm.VarHalfCauchy

    ~liesel_ptm.ScaleWeibull
    ~liesel_ptm.ScaleInverseGamma
    ~liesel_ptm.ScaleHalfCauchy

    ~liesel_ptm.SymmetricallyBoundedScalar
    ~liesel_ptm.TransformedVar

.. rubric:: Data generation

.. autosummary::
    :toctree: generated
    :caption: Data generation
    :recursive:
    :nosignatures:

    ~liesel_ptm.PTMLocScaleDataGen
    ~liesel_ptm.sample_shape

.. rubric:: Helpers

.. autosummary::
    :toctree: generated
    :caption: Helpers
    :recursive:
    :nosignatures:

    ~liesel_ptm.nullspace_remover
    ~liesel_ptm.sumzero
    ~liesel_ptm.diffpen
    ~liesel_ptm.sumzero_term
    ~liesel_ptm.sumzero_coef
    ~liesel_ptm.bspline_basis
    ~liesel_ptm.kn



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
