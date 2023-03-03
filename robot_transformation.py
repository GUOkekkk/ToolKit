# This is a tool kit of some commen transformations in robotics, cause some time I am very confused about it...
# Created by Ke GUO
# ! we always keep the pose form as  [tx, ty, tz, rx, ry, rz, w] in size <1, 7> or <nums, 7>
import numpy as np
from scipy.spatial.transform import Rotation as R
import copy


# define a function to homogenize the coordinate
def homogenize(points: np.ndarray) -> np.ndarray:
    """
    Convert points to homogeneous coordinate

    Args:
        points (np.ndarray): 2D or 3D coordinate <float: num_points, num_dim>

    Returns:
        np.ndarray: 3D or 4D homogeneous coordinate <float: num_points, num_dim + 1>
    """
    if points.ndim == 1:
        return np.append(points, 1)
    else:
        return np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)


# define a function to multiply two quaternions
def quaternion_multiply(quaternion0: np.ndarray, quaternion1: np.ndarray) -> np.ndarray:
    """
     Multiply two quaternions

    Args:
        quaternion0 (np.ndarray):  <4, >
        quaternion1 (np.ndarray):  <4, >

    Returns:
        np.ndarray:  <1, 4>
    """
    _x0, _y0, _z0, _w0 = quaternion0
    _x1, _y1, _z1, _w1 = quaternion1

    ans = np.array(
        [
            _w0 * _x1 + _x0 * _w1 + _y0 * _z1 - _z0 * _y1,
            _w0 * _y1 - _x0 * _z1 + _y0 * _w1 + _z0 * _x1,
            _w0 * _z1 + _x0 * _y1 - _y0 * _x1 + _z0 * _w1,
            _w0 * _w1 - _x0 * _x1 - _y0 * _y1 - _z0 * _z1,
        ],
        dtype=np.float64,
    ).T

    return ans


# define a function to inverse a quaternion
def quaternion_inverse(quaternion: np.ndarray) -> np.ndarray:
    """
       Inverse a quaternion

    Args:
        quaternion (np.ndarray):  <4, >

    Returns:
        the inverse of the quaternion  <4, >
    """
    x0, y0, z0, w0 = quaternion
    return np.array([-x0, -y0, -z0, w0]) / (w0 * w0 + x0 * x0 + y0 * y0 + z0 * z0)


# define a function to get the transformation matrix from the pose
def get_T_from_pose(camera_pose: np.ndarray = np.asarray([0, 0, 0, 0, 0, 0, 1])) -> np.ndarray:
    """
    Get the transformation matrix wTc from the pose

    Args:
        _camera_pose (np.ndarray, optional): Defaults to np.asarray([0, 0, 0, 0, 0, 0, 1]).

    Returns:
        _wTc (np.ndarray): transformation matrix wTc <4, 4>
    """
    _t, _R_quaternion = camera_pose[:3], camera_pose[3:]
    _wRc_mat = R.from_quat(_R_quaternion).as_matrix()

    wTc = np.eye(4)
    wTc[:3, :3] = _wRc_mat
    wTc[:3, 3] = _t.T

    return wTc


# define a function to get the inverse of a transformation matrix
def inverse_transformation(T: np.ndarray) -> np.ndarray:
    """
     Get the inverse of the transformation

    Args:
        _T (np.ndarray): <4, 4>

    Returns:
        _T_inv np.ndarray: <4, 4>
    """
    _R = T[:3, :3]
    _t = T[:3, 3]

    T_inv = np.eye(4)
    T_inv[:3, :3] = _R.T
    T_inv[:3, 3] = -(_R.T).dot(_t).ravel()

    return T_inv


# define a function to move the pose sequence to a given frame,
# we assume the input is a sequence of the pose, maybe it works not good for the iterator
def move_pose(pose_sequence: np.ndarray, given_frame: np.ndarray):
    """
    Move the pose sequence to a given frame
    ! the process of the quaternion
    Args:
        _pose_sequence (np.ndarray): _description_ <nums, 7> rotaiion vector in quaternion form
        _given_frame (np.ndarray): _description_ <nums, 7> rotaiion vector in quaternion form

    Returns:
        _pose_sequence_new (np.ndarray): _description_ <nums, 7> rotaiion vector in quaternion form
    """
    _pose_sequence_new = copy.deepcopy(pose_sequence)

    # move the orientation
    for i in range(len(pose_sequence)):
        _pose_sequence_new[i, 3:] = quaternion_multiply(pose_sequence[i, 3:], quaternion_inverse(given_frame[3:]))

    _wTc0 = np.eye(4)
    _wTc0[:3, :3] = R.from_quat(given_frame[3:]).as_matrix()
    _wTc0[:3, 3] = given_frame[:3].T

    # move the position
    _c0Tw = inverse_transformation(_wTc0)  # Using the np.linalg.inv does not work well

    _t_sequence = _c0Tw.dot(homogenize(_pose_sequence_new[:, :3]).T).T

    return np.hstack((_t_sequence[:, :3], _pose_sequence_new[:, 3:]))


# define a function to transform the line to the matrix
def line2mat(line_data):
    """
    define a function to transform the line to the matrix
    Args:
        line_data (_type_): _description_

    Returns:
        _type_: _description_
    """
    mat = np.eye(4)
    mat[0:3, :] = line_data.reshape(3, 4)
    return np.matrix(mat)


# define a function to transform the motion to the pose which are all in matrix form
def motion2pose(data):
    """
    define a function to transform the motion to the pose which are all in matrix form

    Args:
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    data_size = len(data)
    all_pose = []  # np.zeros((data_size+1, 4, 4))
    all_pose.append(np.eye(4, 4))  # [0,:] = np.eye(4,4)
    pose = np.eye(4, 4)
    for i in range(0, data_size):
        pose = pose.dot(data[i])
        all_pose.append(pose)
    return all_pose


# define a function to transform the relative pose the absolute pose which are all in pose vector 7 form
def rel2abs(data):
    """
    define a function to transform the relative pose the absolute pose which are all in pose vector 7 form

    Args:
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    data_size = len(data)
    all_pose = data[0]  # [0,:] = data[0] # <n, 7>
    pose = pos_quat2SE(data[0])
    for i in range(1, data_size):
        pose = pose.dot(pos_quat2SE(data[i]))
        all_pose = np.vstack((all_pose, SE2pos_quat(pose)))
    return all_pose


# define a function to transform the pose to the motion which are all in matrix form
def pose2motion(data):
    """
    define a function to transform the pose to the motion which are all in matrix form

    Args:
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    data_size = len(data)
    all_motion = []
    for i in range(0, data_size - 1):
        motion = np.linalg.inv(data[i]).dot(data[i + 1])
        all_motion.append(motion)

    return np.array(all_motion)  # N x 4 x 4


# define a function to transform the pose from the matrix form to the 6 vector form
def SE2vect(SE_data):
    """
    define a function to transform the pose from the matrix form to the 6 vector form
    Args:
        SE_data (_type_): _description_

    Returns:
        _type_: _description_
    """
    result = np.zeros((6))
    result[0:3] = np.array(SE_data[0:3, 3].T)
    result[3:6] = SO2rv(SE_data[0:3, 0:3]).T
    return result


# define a function to transform the rotate matrix to the rotate vector
def SO2rv(SO_data):
    """
    define a function to transform the rotate matrix to the rotate vector
    Args:
        SO_data (_type_): _description_

    Returns:
        _type_: _description_
    """
    return R.from_matrix(SO_data).as_rotvec()


# define a function to transform the rotate vector to the rotate matrix
def rv2SO(so_data):
    """
    define a function to transform the rotate vector to the rotate matrix
    Args:
        so_data (_type_): _description_

    Returns:
        _type_: _description_
    """
    return R.from_rotvec(so_data).as_matrix()


# define a function to transform the pose vector to the pose matrix
def vect2SE(se_data):
    """
    define a function to transform the pose vector to the pose matrix
    Args:
        se_data (_type_): _description_

    Returns:
        _type_: _description_
    """
    result_mat = np.matrix(np.eye(4))
    result_mat[0:3, 0:3] = rv2SO(se_data[3:6])
    result_mat[0:3, 3] = np.matrix(se_data[0:3]).T
    return result_mat


# define a function to get the mean value of the pose vector
# ! can get wrong result
def se_mean(se_datas):
    """
    define a function to get the mean value of the pose vector

    Args:
        se_datas (_type_): _description_

    Returns:
        _type_: _description_
    """
    all_SE = np.matrix(np.eye(4))
    for i in range(se_datas.shape[0]):
        se = se_datas[i, :]
        SE = vect2SE(se)
        all_SE = all_SE * SE
    all_se = SE2vect(all_SE)
    mean_se = all_se / se_datas.shape[0]
    return mean_se


def ses_mean(se_datas):
    se_datas = np.array(se_datas)
    se_datas = np.transpose(
        se_datas.reshape(se_datas.shape[0], se_datas.shape[1], se_datas.shape[2] * se_datas.shape[3]), (0, 2, 1)
    )
    se_result = np.zeros((se_datas.shape[0], se_datas.shape[2]))
    for i in range(0, se_datas.shape[0]):
        mean_se = se_mean(se_datas[i, :, :])
        se_result[i, :] = mean_se
    return se_result


def ses2poses(data):
    data_size = data.shape[0]
    all_pose = np.zeros((data_size + 1, 12))
    temp = np.eye(4, 4).reshape(1, 16)
    all_pose[0, :] = temp[0, 0:12]
    pose = np.matrix(np.eye(4, 4))
    for i in range(0, data_size):
        data_mat = vect2SE(data[i, :])
        pose = pose * data_mat
        pose_line = np.array(pose[0:3, :]).reshape(1, 12)
        all_pose[i + 1, :] = pose_line
    return all_pose


def SEs2ses(motion_data):
    data_size = motion_data.shape[0]
    ses = np.zeros((data_size, 6))
    for i in range(0, data_size):
        SE = np.matrix(np.eye(4))
        SE[0:3, :] = motion_data[i, :].reshape(3, 4)
        ses[i, :] = SE2vect(SE)
    return ses


# define a function to transform the rotation vector to the quaternion
def rv2quat(so_data):
    """
    define a function to transform the rotation vector to the quaternion
    Args:
        so_data (_type_): _description_

    Returns:
        _type_: _description_
    """
    so_data = np.array(so_data)
    theta = np.sqrt(np.sum(so_data * so_data))
    axis = so_data / theta
    quat = np.zeros(4)
    quat[0:3] = np.sin(theta / 2) * axis
    quat[3] = np.cos(theta / 2)
    return quat


# define a function to transform the quaternion to rotation vector
def quat2rv(quat_data):
    """
    define a function to transform the quaternion to rotation vector
    Args:
        quat_data (_type_): _description_

    Returns:
        _type_: _description_
    """
    quat_data = np.array(quat_data)
    sin_half_theta = np.sqrt(np.sum(quat_data[0:3] * quat_data[0:3]))
    axis = quat_data[0:3] / sin_half_theta
    cos_half_theta = quat_data[3]
    theta = 2 * np.arctan2(sin_half_theta, cos_half_theta)
    so = theta * axis
    return so


# input so_datas batch*channel*height*width
# return quat_datas batch*numner*channel
def sos2quats(so_datas, mean_std=[[1], [1]]):
    so_datas = np.array(so_datas)
    so_datas = so_datas.reshape(so_datas.shape[0], so_datas.shape[1], so_datas.shape[2] * so_datas.shape[3])
    so_datas = np.transpose(so_datas, (0, 2, 1))
    quat_datas = np.zeros((so_datas.shape[0], so_datas.shape[1], 4))
    for i_b in range(0, so_datas.shape[0]):
        for i_p in range(0, so_datas.shape[1]):
            so_data = so_datas[i_b, i_p, :]
            quat_data = rv2quat(so_data)
            quat_datas[i_b, i_p, :] = quat_data
    return quat_datas


# define a function to transform the rotation matrix to the quaternion
def SO2quat(SO_data):
    """
    define a function to transform the rotation matrix to the quaternion
    Args:
        SO_data (_type_): _description_

    Returns:
        _type_: _description_
    """
    rr = R.from_matrix(SO_data)
    return rr.as_quat()


# define a function to transform the quaternion to the rotation matrix
def quat2SO(quat_data):
    """
    define a function to transform the quaternion to the rotation matrix

    Args:
        quat_data (_type_): _description_

    Returns:
        _type_: _description_
    """
    return R.from_quat(quat_data).as_matrix()


# define a function to transform the pose vector(1, 7) to the pose matrix but in line
def pos_quat2SE_vector(quat_data):
    """
    define a function to transform the pose vector(1, 7) to the pose matrix but in line

    Args:
        quat_data (_type_): _description_

    Returns:
        _type_: _description_
    """
    SO = R.from_quat(quat_data[3:7]).as_matrix()
    SE = np.matrix(np.eye(4))
    SE[0:3, 0:3] = np.matrix(SO)
    SE[0:3, 3] = np.matrix(quat_data[0:3]).T
    SE = np.array(SE[0:3, :]).reshape(1, 12)
    return SE


# define a function to transform the pose vector(1, 7) to the pose matrix
def pos_quat2SE(quat_data):
    """
    define a function to transform the pose vector(1, 7) to the pose matrix

    Args:
        quat_data (_type_): _description_

    Returns:
        _type_: _description_
    """
    SO = R.from_quat(quat_data[3:7]).as_matrix()
    SE = np.matrix(np.eye(4))
    SE[0:3, 0:3] = np.matrix(SO)
    SE[0:3, 3] = np.matrix(quat_data[0:3]).T
    return SE


def pos_quats2SE_vectors(quat_datas):
    data_len = quat_datas.shape[0]
    SEs = np.zeros((data_len, 12))
    for i_data in range(0, data_len):
        SE = pos_quat2SE(quat_datas[i_data, :])
        SEs[i_data, :] = SE
    return SEs


# define a function to transform the pose vector(1, 7) to the pose matrix
def pos_quats2SE(quat_datas):
    """
    define a function to transform the pose vector(1, 7) to the pose matrix
    Args:
        quat_datas (_type_): _description_

    Returns:
        _type_: _description_
    """
    SEs = []
    for quat in quat_datas:
        SO = R.from_quat(quat[3:7]).as_matrix()
        SE = np.eye(4)
        SE[0:3, 0:3] = SO
        SE[0:3, 3] = quat[0:3]
        SEs.append(SE)
    return SEs


# define a function to transform the the pose matrix to the pose vector(1, 7)
def SE2pos_quat(SE_data):
    """
    define a function to transform the the pose matrix to the pose vector(1, 7)
    Args:
        SE_data (_type_): _description_

    Returns:
        _type_: _description_
    """
    pos_quat = np.zeros(7)
    pos_quat[3:] = SO2quat(SE_data[0:3, 0:3])
    pos_quat[:3] = SE_data[0:3, 3].T
    return pos_quat


# define a function to transform the camera trajectory from the NED frame to the camera frame
def NED2camera(pose):
    T = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
    T_inv = np.linalg.inv(T)
    pose_mat = pos_quat2SE(pose)
    return SE2pos_quat(T @ pose_mat @ T_inv)


# define a function to transform the camera trajectory from the camera frame to the NED frame
def camera2NED(pose):
    T = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
    T_inv = np.linalg.inv(T)
    pose_mat = pos_quat2SE(pose)
    return SE2pos_quat(T @ pose_mat @ T_inv)


# define a function to get the relative pose of pose 1 in pose 0 frame
def get_relative_pose(pose0: np.ndarray, pose1: np.ndarray):
    """
    define a function to get the relative pose of pose 1 in pose 0 frame
    Args:
        pose0 (np.ndarray): _description_
        pose1 (np.ndarray): _description_

    Returns:
        _type_: _description_
    """
    pose0_mat = pos_quat2SE(pose0)
    pose1_mat = pos_quat2SE(pose1)
    relative_pose_mat = inverse_transformation(pose0_mat) @ pose1_mat @ pose0_mat
    relative_pose = SE2pos_quat(relative_pose_mat)
    return relative_pose


# define a function to move the rotation along x and z axis of the camera, cause we condier the motion is a 2D motion,
# we assume the input is a sequence of the pose, maybe it works not good for the iterator
def remove_rotation_xz(pose_sequence: np.ndarray) -> np.ndarray:
    """
     Move the useless rotation from the pose

    Args:
        _pose_sequence (np.ndarray): <nums, 7> we assume the input is the whole pose

    Returns:
        _pose_sequence_new (np.ndarray): _description_ <nums, 7> rotaiion vector in quaternion form
    """
    pose_sequence_new = copy.deepcopy(pose_sequence)
    for i in range(len(pose_sequence)):
        rx, ry, rz = R.from_quat(pose_sequence[i, 3:]).as_euler("xyz", degrees=False)
        pose_sequence_new[i, 3:] = R.from_euler("xyz", [0, ry, 0], degrees=False).as_quat()

    return pose_sequence_new
