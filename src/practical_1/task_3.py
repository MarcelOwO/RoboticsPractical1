import numpy as np
from .utility import *
from .task_1 import *
from .task_2 import *
from .task_3 import *


def run_task_3():
    print("running task 3")
    check_task_3()


def euler_zyx(C: np.ndarray) -> tuple[float, float, float]:
    """Return (z, y, x) for C = Cz(z) @ Cy(y) @ Cx(x), |y| < pi/2.

    Use atan2 for each angle and hypot(C[2,1], C[2,2]) for |cos(y)|.
    Raise ValueError if |cos(y)| < 1e-8: at gimbal lock z and x cannot be
    recovered separately. The matrix remains a valid orientation.
    We deliberately expose the singular case instead of assigning an arbitrary
    yaw/roll pair. Other branches are outside this exercise.
    """

    C = np.asarray(C, dtype=float)

    if C.shape != (3, 3):
        raise ValueError("C is of the wrong shape")

    cos_y = np.hypot(C[2, 1], C[2, 2])

    if cos_y < 1e-8:
        raise ValueError("Gimbal lock")

    y = np.arctan2(-C[2, 0], cos_y)
    x = np.arctan2(C[2, 1], C[2, 2])
    z = np.arctan2(C[1, 0], C[0, 0])

    return (z, y, x)


def orientation_error_world(C: np.ndarray, C_goal: np.ndarray) -> np.ndarray:
    """World-axis rotation vector phi satisfying exp_so3(phi) @ C == C_goal.

    Form the relative rotation first, then use the supplied log_so3.
    The principal result has norm <= pi; its axis sign is ambiguous at pi.
    """

    C = np.asarray(C, dtype=float)
    C_goal = np.asarray(C_goal, dtype=float)

    if C.shape != (3, 3):
        raise ValueError("Invalid shape for C")

    if C_goal.shape != (3, 3):
        raise ValueError("Invalid shape for C_goal")

    C_rel = C_goal @ C.T

    return log_so3(C_rel)


def slerp(q0: np.ndarray, q1: np.ndarray, s: float) -> np.ndarray:
    """Unit quaternion at fraction s in [0,1] along a shortest rotation path.

    Inputs are unit quaternions, scalar-first. If dot(q0,q1) < 0, flip q1.
    Clip the dot product to [-1,1] before arccos. For dot > 1-1e-8, use
    normalized linear interpolation; otherwise use the sine-weighted formula.
    Normalize the output to remove roundoff. At a relative half-turn, either
    shortest path is valid; the input signs select one.
    """
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)

    # if q0.shape != (3, 3):
    # raise ValueError(f"shape of q0 is not (3,3) -> '{q0.shape}'.")

    # if q1.shape != (3, 3):
    # raise ValueError(f"shape of q1 is not (3,3) -> '{q1.shape}'.")

    dot = np.dot(q0, q1)

    if dot < 0:
        q1 = -q1
        dot = -dot

    dot = np.clip(dot, -1.0, 1.0)

    if dot > 1.0 - 1e-8:
        res = (1.0 - s) * q0 + s * q1

    else:
        theta = np.acos(dot)
        sin_theta = np.sin(theta)
        res = (np.sin((1.0 - s) * theta) / sin_theta) * q0 + (
            np.sin(s * theta) / sin_theta
        ) * q1

    return res / np.linalg.norm(res)


def check_task_3():
    for pitch in [-100, -89.9, 0, 40, 89.9, 90.1]:
        C = Cz(0.4) @ Cy(np.deg2rad(pitch)) @ Cx(-0.2)
        z, y, x = euler_zyx(C)
        assert abs(y) < np.pi / 2
        assert np.allclose(Cz(z) @ Cy(y) @ Cx(x), C)
    for pitch in [-90, 90]:
        C = Cz(0.4) @ Cy(np.deg2rad(pitch)) @ Cx(-0.2)
        assert is_rotation(C)
        try:
            euler_zyx(C)
        except ValueError:
            pass
        else:
            raise AssertionError("Gimbal lock must raise ValueError")
    C = Cy(0.7) @ Cx(0.4)
    wanted = np.array([0.2, -0.1, 0.3])
    goal = exp_so3(wanted) @ C
    error = orientation_error_world(C, goal)
    assert np.allclose(error, wanted)
    assert np.allclose(exp_so3(error) @ C, goal)
    q0 = quat_from_phi(np.deg2rad([0, 0, 170]))
    q1 = quat_from_phi(np.deg2rad([0, 0, -170]))
    error = orientation_error_world(C_from_quat(q0), C_from_quat(q1))
    assert np.allclose(error, [0, 0, np.deg2rad(20)])
    for s in [0, 0.25, 0.5, 0.75, 1]:
        for end in [q1, -q1]:
            q = slerp(q0, end, s)
            assert np.isclose(np.linalg.norm(q), 1)
            assert np.allclose(C_from_quat(q), Cz(np.deg2rad(170 + 20 * s)))
        for end in [q0, -q0]:
            q = slerp(q0, end, s)
            assert np.all(np.isfinite(q))
            assert np.isclose(np.linalg.norm(q), 1)
            assert np.allclose(C_from_quat(q), C_from_quat(q0))
    print("Task 3 checks passed")
