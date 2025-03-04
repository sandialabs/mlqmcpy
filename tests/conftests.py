import numpy as np
import pytest


@pytest.fixture(autouse=True)
def set_random_seed():
    """Automatically sets a fixed random seed before each test."""
    np.random.seed(1234)
