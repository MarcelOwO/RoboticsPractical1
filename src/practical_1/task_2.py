import numpy as np
from .task_1 import is_rotation
from .utility import *


def run_task_2():
    print("running task 2")
    check_task_2()


def exp_so3(phi: np.ndarray) -> np.ndarray:
    """Rodrigues: C = exp([phi]_x) = I + sin(t)[n]_x + (1 - cos(t))[n]_x^2, t = |phi|."""

    phi = np.asarray(phi, dtype=float)

    if phi.shape != (3,):
        raise ValueError("Expected phi to be of Shape (3,).")

    t = np.linalg.norm(phi)

    if np.isclose(t, 0.0):
        return np.eye(3)

    n = phi / t

    I = np.eye(3)

    N = skew(n)

    return I + np.sin(t) * N + (1 - np.cos(t)) * (N @ N)


def check_task_2():
    assert np.allclose(exp_so3(np.zeros(3)), np.eye(3))
    rng = np.random.default_rng(560)
    for angle in [1e-8, 1e-3, 1.0, np.pi - 1e-3, np.pi]:
        axis = rng.normal(size=3)
        phi = angle * axis / np.linalg.norm(axis)
        C = exp_so3(phi)
        q = quat_from_phi(phi)
        assert is_rotation(C)
        assert np.allclose(C, C_from_quat(q), atol=1e-7)
        assert np.allclose(exp_so3(log_so3(C)), C, atol=1e-7)
        assert np.allclose(C_from_quat(q), C_from_quat(-q))
        assert np.allclose(C_from_quat(quat_from_C(C)), C, atol=1e-7)
    print("Task 2 checks passed")
