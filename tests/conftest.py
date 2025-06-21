import numpy as np
import pytest


@pytest.fixture
def check_dicts():
    def _method(actual: dict, expected: dict, rtol=1e-4):
        assert actual.keys() == expected.keys()
        for k, v in expected.items():
            np.testing.assert_allclose(actual[k], v, rtol=rtol)
    return _method
