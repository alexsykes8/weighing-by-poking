"""Rest-bend removal for real clips: the shape over the first still window is subtracted, axial x untouched."""
import numpy as np

from probe.estimator import zero_rest_shape


def test_rest_bend_removed():
    t = np.arange(60) / 30.0
    s = np.linspace(0, 0.33, 9)
    cl = np.zeros((60, 9, 3))
    cl[:, :, 0] = s
    rest = -0.003 * (s / s[-1]) ** 2                     # 3 mm tip offset before any load
    load = np.where(t > 1.0, -0.02, 0.0)[:, None] * (s / s[-1]) ** 2
    cl[:, :, 2] = rest + load
    out = zero_rest_shape(t, cl)
    np.testing.assert_allclose(out[:, :, 2], load, atol=1e-12)
    np.testing.assert_array_equal(out[:, :, 0], cl[:, :, 0])
    assert cl[0, -1, 2] == rest[-1]                      # input not modified
