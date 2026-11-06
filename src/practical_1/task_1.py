from .utility import *


def run_task_1():
    print("Running task 1")
    check_task_1()
    return


def C_from_rpy(rpy: list[float]) -> np.ndarray:
    """Rotation matrix for fixed-axis roll, pitch, yaw (radians).

    Turn about the FIXED axes x, then y, then z, which is the same matrix
    as Cz(yaw) @ Cy(pitch) @ Cx(roll).
    """
    if len(rpy) != 3:
        raise ValueError("Expected exactly 3 values")

    roll, pitch, yaw = rpy[0], rpy[1], rpy[2]

    return Cz(yaw) @ Cy(pitch) @ Cx(roll)


def is_rotation(C: np.ndarray, tol=1e-9) -> bool:
    """True if C is in SO(3): C^T C = I and det C = +1 (up to tol)."""

    if C.shape != (3, 3):
        raise ValueError("Expected exactly 3 values")
    identity = np.eye(3)

    orth = np.allclose(C.T @ C, identity, atol=tol, rtol=0)
    det = np.isclose(np.linalg.det(C), 1.0, atol=tol, rtol=0)

    return bool(orth and det)


def check_task_1():
    assert np.allclose(Cz(np.pi / 2) @ [1, 0, 0], [0, 1, 0])
    C = C_from_rpy([np.pi / 2, 0, np.pi / 2])
    assert np.allclose(C @ [0, 1, 0], [0, 0, 1])
    assert np.allclose(C @ [1, 0, 0], [0, 1, 0])
    A, B = Cz(np.pi / 2), Cx(np.pi / 2)
    assert not np.allclose(A @ B, B @ A)
    assert np.allclose((A @ B).T, B.T @ A.T)
    assert np.allclose(C.T @ C, np.eye(3))
    assert is_rotation(C)
    for invalid in [
        np.diag([1.0, 1.0, -1.0]),
        1.01 * np.eye(3),
        np.array([[1, -0.1, 0], [0.1, 1, 0], [0, 0, 1]]),
    ]:
        assert not is_rotation(invalid)
    print("Task 1 checks passed")
