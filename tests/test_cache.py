import time
import timeit

import dill
import tensorflow_probability.substrates.jax.bijectors as tfb

from liesel_ptm import cache
from liesel_ptm.var import InvertBijector


def test_cache_returns_same_result_if_appropriate(tmp_path):
    # Define a test function to be cached
    @cache(directory=tmp_path)
    def test_function(arg):
        return arg * 2

    result1 = test_function(5)
    result2 = test_function(5)
    assert result1 == result2


def test_cache_returns_new_result_if_appropriate(tmp_path):
    # Define a test function to be cached
    @cache(directory=tmp_path)
    def test_function(arg):
        return arg * 2

    result1 = test_function(5)
    result2 = test_function(4)
    assert result1 == 5 * 2
    assert result2 == 4 * 2


def test_cache_file_is_created(tmp_path):
    # Define a test function to be cached
    @cache(directory=tmp_path)
    def test_function(arg):
        return arg * 2

    result1 = test_function(5)
    fname = next(tmp_path.iterdir())

    with open(fname, "rb") as file:
        result = dill.load(file)

    assert fname.exists()
    assert result1 == result


def test_cache_improves_speed(tmp_path):
    @cache(directory=tmp_path)
    def test_function():
        time.sleep(0.1)
        return 2

    res1 = timeit.timeit(test_function, number=1)
    res2 = timeit.timeit(test_function, number=1)

    assert res2 < res1


def test_cache_not_as_decorator(tmp_path):
    def test_function(arg):
        return arg * 2

    result1 = cache(tmp_path)(test_function)(5)
    result2 = cache(tmp_path)(test_function)(5)
    assert result1 == result2


def test_cache_bijector(tmp_path):
    @cache(tmp_path)
    def bijector():
        return tfb.Exp()

    bijector()
    bijector()


def test_cache_inverse_bijector(tmp_path):
    @cache(tmp_path)
    def bijector():
        return InvertBijector(bijector=tfb.Exp())

    bijector()
    bijector()
