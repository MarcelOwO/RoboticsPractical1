import numpy as np
import matplotlib.pyplot as plt

EPS = 1e-9


def skew(a):
    """[a]_x, the matrix with skew(a) @ b == np.cross(a, b)."""
    return np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])


def vee(S):
    """Inverse of skew: the vector a with skew(a) == S (S skew-symmetric)."""
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def Cx(a):
    """Elementary rotation about x by the angle a (rad)."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def Cy(a):
    """Elementary rotation about y by the angle a (rad)."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def Cz(a):
    """Elementary rotation about z by the angle a (rad)."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def log_so3(C):
    """Rotation vector phi with exp_so3(phi) == C, angle in [0, pi].

    Use theta = atan2(|vee(C - C^T)| / 2, (tr C - 1) / 2). Near theta = pi
    the axis must come from the symmetric part of C instead.

    Returns a principal rotation vector. Its axis sign can be ambiguous
    at a half-turn.
    """
    C = np.asarray(C, float)
    w = 0.5 * vee(C - C.T)  # = sin(theta) n
    s = np.linalg.norm(w)
    c = 0.5 * (np.trace(C) - 1.0)  # = cos(theta)
    theta = np.arctan2(s, c)
    if theta < 1e-6:
        return w  # small angle: phi ~ vee(C - C^T)/2
    if np.pi - theta < 1e-4:
        # C ~ 2 n n^T - I: take the largest column of (C + I) / 2
        B = 0.5 * (C + np.eye(3))
        k = int(np.argmax(np.diag(B)))
        n = B[:, k] / np.sqrt(B[k, k])
        if n @ w < 0:  # keep the sign consistent with w
            n = -n
        return theta * n / np.linalg.norm(n)
    return theta * w / s


def quat_mul(p, q):
    """Hamilton product p (x) q."""
    p0, pv = p[0], np.asarray(p[1:], float)
    q0, qv = q[0], np.asarray(q[1:], float)
    return np.r_[p0 * q0 - pv @ qv, p0 * qv + q0 * pv + np.cross(pv, qv)]


def quat_conj(q):
    """Conjugate (w, -x, -y, -z); the inverse of a unit quaternion."""
    return np.r_[q[0], -np.asarray(q[1:], float)]


def quat_from_phi(phi):
    """Unit quaternion (cos(t/2), sin(t/2) n) of the rotation vector phi = t n."""
    phi = np.asarray(phi, float)
    t = np.linalg.norm(phi)
    if t < EPS:
        return np.r_[1.0, 0.5 * phi]
    return np.r_[np.cos(t / 2), np.sin(t / 2) * phi / t]


def C_from_quat(q):
    """Rotation matrix of a **unit** quaternion: (2 q0^2 - 1) I + 2 q0 [qv]_x + 2 qv qv^T."""
    q0, qv = q[0], np.asarray(q[1:], float)
    return (2 * q0**2 - 1) * np.eye(3) + 2 * q0 * skew(qv) + 2 * np.outer(qv, qv)


def quat_from_C(C):
    """Unit quaternion of a rotation matrix, with w >= 0 (one of the two choices +-q)."""
    return quat_from_phi(log_so3(C))
