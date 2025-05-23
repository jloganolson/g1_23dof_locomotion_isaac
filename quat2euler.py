import numpy as np

# --- Copied functions from mujoco_worldgen/util/rotation.py ---
# For testing whether a number is close to zero
_FLOAT_EPS = np.finfo(np.float64).eps
_EPS4 = _FLOAT_EPS * 4.0

def mat2euler(mat):
    """ Convert Rotation Matrix to Euler Angles. See rotation.py for notes """
    mat = np.asarray(mat, dtype=np.float64)
    assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)

    cy = np.sqrt(mat[..., 2, 2] * mat[..., 2, 2] + mat[..., 1, 2] * mat[..., 1, 2])
    condition = cy > _EPS4
    euler = np.empty(mat.shape[:-1], dtype=np.float64)
    euler[..., 2] = np.where(condition,
                             -np.arctan2(mat[..., 0, 1], mat[..., 0, 0]),
                             -np.arctan2(-mat[..., 1, 0], mat[..., 1, 1]))
    euler[..., 1] = np.where(condition,
                             -np.arctan2(-mat[..., 0, 2], cy),
                             -np.arctan2(-mat[..., 0, 2], cy))
    euler[..., 0] = np.where(condition,
                             -np.arctan2(mat[..., 1, 2], mat[..., 2, 2]),
                             0.0)
    return euler


def quat2mat(quat):
    """ Convert Quaternion to Euler Angles. See rotation.py for notes """
    quat = np.asarray(quat, dtype=np.float64)
    assert quat.shape[-1] == 4, "Invalid shape quat {}".format(quat)

    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    Nq = np.sum(quat * quat, axis=-1)
    s = 2.0 / Nq
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z

    mat = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 0, 0] = 1.0 - (yY + zZ)
    mat[..., 0, 1] = xY - wZ
    mat[..., 0, 2] = xZ + wY
    mat[..., 1, 0] = xY + wZ
    mat[..., 1, 1] = 1.0 - (xX + zZ)
    mat[..., 1, 2] = yZ - wX
    mat[..., 2, 0] = xZ - wY
    mat[..., 2, 1] = yZ + wX
    mat[..., 2, 2] = 1.0 - (xX + yY)
    return np.where((Nq > _FLOAT_EPS)[..., np.newaxis, np.newaxis], mat, np.eye(3))


def quat2euler(quat):
    """ Convert Quaternion to Euler Angles. See rotation.py for notes """
    return mat2euler(quat2mat(quat))

def euler2quat(euler):
    """ Convert Euler Angles to Quaternions. See rotation.py for notes """
    euler = np.asarray(euler, dtype=np.float64)
    assert euler.shape[-1] == 3, "Invalid shape euler {}".format(euler)

    # Note: The original code in rotation.py has specific conventions for
    # how Euler angles are applied to match MuJoCo's default 'xyz' relative rotating frame.
    # The signs for ai, aj, ak are crucial here.
    ai, aj, ak = euler[..., 2] / 2, -euler[..., 1] / 2, euler[..., 0] / 2
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    quat = np.empty(euler.shape[:-1] + (4,), dtype=np.float64)
    quat[..., 0] = cj * cc + sj * ss # w
    quat[..., 3] = cj * sc - sj * cs # z
    quat[..., 2] = -(cj * ss + sj * cc) # y
    quat[..., 1] = cj * cs - sj * sc # x
    return quat

# --- End of copied functions ---


# A sample quaternion (w, x, y, z)
# This quaternion represents a rotation around the X-axis by approximately 90 degrees
q = np.array([0.999803, -6.03319e-05, 0.0198256, 0.00131986])

# Convert quaternion to Euler angles (roll, pitch, yaw in radians)
# This uses the copied quat2euler function
euler_angles_rad = quat2euler(q)

# Convert radians to degrees for easier interpretation
euler_angles_deg = np.degrees(euler_angles_rad)

print(f"Quaternion: {q}")
print(f"Euler Angles (radians - ZYX): {euler_angles_rad}")
print(f"Euler Angles (degrees - ZYX): {euler_angles_deg}")

# --- Optional: Convert Euler angles back to quaternion to verify ---
# This uses the copied euler2quat function
q_reconverted = euler2quat(euler_angles_rad)
print(f"Reconverted Quaternion: {q_reconverted}")

# You can also use the identity quaternion (no rotation)
q_identity = np.array([1.0, 0.0, 0.0, 0.0])
euler_identity_rad = quat2euler(q_identity)
print(f"\nIdentity Quaternion: {q_identity}")
print(f"Euler Angles (identity - degrees): {np.degrees(euler_identity_rad)}")