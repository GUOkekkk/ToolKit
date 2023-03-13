# This is a evaluatioin/metrics tool of SLAM/VO performance, it contains the common metrics ATE, RPE, KITTI score
# and one customized metric called the Success Rate to check the performance of the algorithm on one trajectory
# Created by Ke GUO in 13/03/2023

import numpy as np
from scipy.spatial.transform import Rotation as R
import robot_transformation as robT
import random

# We assume the input pose or trajectory should be in the form of <n, 7> which means the rotation in the quaternion form


# ==============================
# Some useful functions


def trajectory_distances(poses):
    distances = []
    distances.append(0)
    for i in range(1, len(poses)):
        p1 = poses[i - 1]
        p2 = poses[i]
        delta = p1[0:3, 3] - p2[0:3, 3]
        distances.append(distances[i - 1] + np.linalg.norm(delta))
    return distances


def last_frame_from_segment_length(dist, first_frame, length):
    for i in range(first_frame, len(dist)):
        if dist[i] > dist[first_frame] + length:
            return i
    return -1


def rotation_error(pose_error):
    a = pose_error[0, 0]
    b = pose_error[1, 1]
    c = pose_error[2, 2]
    d = 0.5 * (a + b + c - 1)
    rot_error = np.arccos(max(min(d, 1.0), -1.0))
    return rot_error


def translation_error(pose_error):
    dx = pose_error[0, 3]
    dy = pose_error[1, 3]
    dz = pose_error[2, 3]
    return np.sqrt(dx * dx + dy * dy + dz * dz)


def calculate_sequence_error(poses_gt, poses_result, lengths=[10, 20, 30, 40, 50, 60, 70, 80]):
    # error_vetor
    errors = []

    # paramet
    step_size = 1  # 10; # every second
    num_lengths = len(lengths)

    # import ipdb;ipdb.set_trace()
    # pre-compute distances (from ground truth as reference)
    dist = trajectory_distances(poses_gt)  # the Euclidean distance of 2 pose
    # for all start positions do
    for first_frame in range(0, len(poses_gt), step_size):
        # for all segment lengths do
        for i in range(0, num_lengths):
            #  current length
            length = lengths[i]

            # compute last frame
            # Use the length in this part
            last_frame = last_frame_from_segment_length(dist, first_frame, length)
            # continue, if sequence not long enough
            if last_frame == -1:
                continue

            # compute rotational and translational errors
            # Normal, similar to the RPE
            pose_delta_gt = np.linalg.inv(poses_gt[first_frame]).dot(poses_gt[last_frame])
            pose_delta_result = np.linalg.inv(poses_result[first_frame]).dot(poses_result[last_frame])
            pose_error = np.linalg.inv(pose_delta_result).dot(pose_delta_gt)
            r_err = rotation_error(pose_error)
            t_err = translation_error(pose_error)

            # compute speed
            # Use the speed in this part
            num_frames = (float)(last_frame - first_frame + 1)
            speed = length / (0.1 * num_frames)

            # write to file
            error = [first_frame, r_err / length, t_err / length, length, speed]
            errors.append(error)
            # return error vector
    return errors


def calculate_ave_errors(errors, lengths=[10, 20, 30, 40, 50, 60, 70, 80]):
    rot_errors = []
    tra_errors = []
    for length in lengths:
        rot_error_each_length = []
        tra_error_each_length = []
        for error in errors:
            if abs(error[3] - length) < 0.1:
                rot_error_each_length.append(error[1])
                tra_error_each_length.append(error[2])

        if len(rot_error_each_length) == 0:
            # import ipdb;ipdb.set_trace()
            continue
        else:
            rot_errors.append(sum(rot_error_each_length) / len(rot_error_each_length))
            tra_errors.append(sum(tra_error_each_length) / len(tra_error_each_length))
    return np.array(rot_errors) * 180 / np.pi, tra_errors


def kittievaluate(gt, data, rescale_=False):
    lens = [5, 10, 15, 20, 25, 30, 35, 40]  # [1,2,3,4,5,6] #
    errors = calculate_sequence_error(gt, data, lengths=lens)
    rot, tra = calculate_ave_errors(errors, lengths=lens)
    return np.mean(rot), np.mean(tra)


def evaluate_trajectory(traj_gt, traj_est, param_max_pairs=10000, param_fixed_delta=False, param_delta=1.00):
    """
    Compute the relative pose error between two trajectories.

    Input:
    traj_gt -- the first trajectory (ground truth)
    traj_est -- the second trajectory (estimated trajectory)
    param_max_pairs -- number of relative poses to be evaluated
    param_fixed_delta -- false: evaluate over all possible pairs
                         true: only evaluate over pairs with a given distance (delta)
    param_delta -- distance between the evaluated pairs
    param_delta_unit -- unit for comparison:
                        "s": seconds
                        "m": meters
                        "rad": radians
                        "deg": degrees
                        "f": frames
    param_offset -- time offset between two trajectories (to model the delay)
    param_scale -- scale to be applied to the second trajectory

    Output:
    list of compared poses and the resulting translation and rotation error
    """

    if not param_fixed_delta:
        if param_max_pairs == 0 or len(traj_est) < np.sqrt(param_max_pairs):
            pairs = [(i, j) for i in range(len(traj_est)) for j in range(len(traj_est))]
        else:
            pairs = [
                (random.randint(0, len(traj_est) - 1), random.randint(0, len(traj_est) - 1))
                for i in range(param_max_pairs)
            ]
    else:
        pairs = []
        for i in range(len(traj_est)):
            j = i + param_delta
            if j < len(traj_est):
                pairs.append((i, j))
        if param_max_pairs != 0 and len(pairs) > param_max_pairs:
            pairs = random.sample(pairs, param_max_pairs)

    result = []
    for i, j in pairs:

        error44 = ominus(ominus(traj_est[j], traj_est[i]), ominus(traj_gt[j], traj_gt[i]))

        trans = compute_distance(error44)
        rot = compute_angle(error44)

        result.append([i, j, trans, rot])

    if len(result) < 2:
        raise Exception("Couldn't find pairs between groundtruth and estimated trajectory!")

    return result


def align(model, data, calc_scale=False):
    """Align two trajectories using the method of Horn (closed-form).

    Input:
    model -- first trajectory (3xn)
    data -- second trajectory (3xn)

    Output:
    rot -- rotation matrix (3x3)
    trans -- translation vector (3x1)
    trans_error -- translational error per point (1xn)

    """
    np.set_printoptions(precision=3, suppress=True)
    model_zerocentered = model - model.mean(1)
    data_zerocentered = data - data.mean(1)

    W = np.zeros((3, 3))
    for column in range(model.shape[1]):
        W += np.outer(model_zerocentered[:, column], data_zerocentered[:, column])
    U, d, Vh = np.linalg.linalg.svd(W.transpose())
    S = np.matrix(np.identity(3))
    if np.linalg.det(U) * np.linalg.det(Vh) < 0:
        S[2, 2] = -1
    rot = U * S * Vh

    if calc_scale:
        rotmodel = rot * model_zerocentered
        dots = 0.0
        norms = 0.0
        for column in range(data_zerocentered.shape[1]):
            dots += np.dot(data_zerocentered[:, column].transpose(), rotmodel[:, column])
            normi = np.linalg.norm(model_zerocentered[:, column])
            norms += normi * normi
        # s = float(dots/norms)
        s = float(norms / dots)
    else:
        s = 1.0

    # trans = data.mean(1) - s*rot * model.mean(1)
    # model_aligned = s*rot * model + trans
    # alignment_error = model_aligned - data

    # scale the est to the gt, otherwise the ATE could be very small if the est scale is small
    trans = s * data.mean(1) - rot * model.mean(1)
    model_aligned = rot * model + trans
    data_alingned = s * data
    alignment_error = model_aligned - data_alingned

    trans_error = np.sqrt(np.sum(np.multiply(alignment_error, alignment_error), 0)).A[0]

    return rot, trans, trans_error, s


# ===============================
# ! ATE metrics
class ATEEvaluator(object):
    def __init__(self):
        super(ATEEvaluator, self).__init__()

    def evaluate(self, gt_traj, est_traj, scale):
        gt_xyz = np.matrix(gt_traj[:, 0:3].transpose())
        est_xyz = np.matrix(est_traj[:, 0:3].transpose())

        rot, trans, trans_error, s = align(gt_xyz, est_xyz, scale)
        print("  ATE scale: {}".format(s))
        error = np.sqrt(np.dot(trans_error, trans_error) / len(trans_error))

        # align two trajs
        est_SEs = robT.pos_quats2SE(est_traj)
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3:] = trans
        T = np.linalg.inv(T)
        est_traj_aligned = []
        for se in est_SEs:
            se[:3, 3] = se[:3, 3] * s
            se_new = T.dot(se)
            se_new = robT.SE2pos_quat(se_new)
            est_traj_aligned.append(se_new)

        return error, gt_traj, est_traj_aligned


# ===============================
# ! RPE metrics
class RPEEvaluator(object):
    def __init__(self):
        super(RPEEvaluator, self).__init__()

    def evaluate(self, gt_SEs, est_SEs):
        result = evaluate_trajectory(gt_SEs, est_SEs)

        trans_error = np.array(result)[:, 2]
        rot_error = np.array(result)[:, 3]

        trans_error_mean = np.mean(trans_error)
        rot_error_mean = np.mean(rot_error)

        # import ipdb;ipdb.set_trace()

        return (rot_error_mean, trans_error_mean)


# ======================================
# ! KITTI metrics
class KittiEvaluator(object):
    def __init__(self):
        super(KittiEvaluator, self).__init__()

    # return rot_error, tra_error
    def evaluate(self, gt_SEs, est_SEs):
        # trajectory_scale(est_SEs, 0.831984631412)
        error = kittievaluate(gt_SEs, est_SEs)
        return error


# ===============================
# ! SR metrics
def ominus(a, b):
    """
    Compute the relative 3D transformation between a and b.

    Input:
    a -- first pose (homogeneous 4x4 matrix)
    b -- second pose (homogeneous 4x4 matrix)

    Output:
    Relative 3D transformation from a to b.
    """
    return np.dot(np.linalg.inv(a), b)


def compute_distance(transform):
    """
    Compute the distance of the translational component of a 4x4 homogeneous matrix.
    """
    return np.linalg.norm(transform[0:3, 3])


def compute_angle(transform):
    """
    Compute the rotation angle from a 4x4 homogeneous matrix.
    """
    # an invitation to 3-d vision, p 27
    return np.arccos(min(1, max(-1, (np.trace(transform[0:3, 0:3]) - 1) / 2)))


class SREvaluator(object):
    def __init__(self):
        super(SREvaluator, self).__init__()

    def calculate_rpe(self, gt_SEs, est_SEs, delta=20):
        rpe = []
        # 0 - delta
        for i in range(0, delta):
            error44 = ominus(ominus(est_SEs[i + delta], est_SEs[i]), ominus(gt_SEs[i + delta], gt_SEs[i]))
            trans = compute_distance(error44)
            rot = compute_angle(error44)
            rpe.append([i, i + delta, rot, trans])

        # delta - len-delta
        for i in range(delta, len(gt_SEs) - delta):
            error44_0 = ominus(ominus(est_SEs[i + delta], est_SEs[i]), ominus(gt_SEs[i + delta], gt_SEs[i]))
            trans_0 = compute_distance(error44_0)
            rot_0 = compute_angle(error44_0)

            error44_1 = ominus(ominus(est_SEs[i - delta], est_SEs[i]), ominus(gt_SEs[i - delta], gt_SEs[i]))
            trans_1 = compute_distance(error44_1)
            rot_1 = compute_angle(error44_1)

            rpe.append([i - delta, i + delta, (rot_0 + rot_1) / 2, (trans_0 + trans_1) / 2])

        for i in range(len(gt_SEs) - delta, len(gt_SEs)):

            error44 = ominus(ominus(est_SEs[i - delta], est_SEs[i]), ominus(gt_SEs[i - delta], gt_SEs[i]))
            trans = compute_distance(error44)
            rot = compute_angle(error44)
            rpe.append([i - delta, i, rot, trans])

        return rpe

    # We do not use the scale here, cause we are doing the PnP here, the scale ambiguity not too serious
    def calculate_ate(self, gt_traj, est_traj):
        t_error = gt_traj[:, :3] - est_traj[:, :3]
        ate = np.sqrt(np.sum(t_error * t_error, axis=1))
        return ate

    def calculate_aoe(self, gt_traj, est_traj):
        aoe = []
        for i in range(len(gt_traj)):
            R_error = np.linalg.inv(R.from_quat(gt_traj[i, 3:]).as_matrix()) @ R.from_quat(est_traj[i, 3:]).as_matrix()
            aoe.append(compute_angle(R_error))
        return np.asarray(aoe)

    def evaluate(self, gt_SEs, est_SEs, gt_traj, est_traj, scale, epsilo, phi, delta):
        # rpe_error = np.asarray(self.calculate_rpe(gt_SEs, est_SEs, delta=delta))
        traj_ate = self.calculate_ate(gt_traj, est_traj)
        traj_aoe = self.calculate_aoe(gt_traj, est_traj)
        traj_s = np.vstack((range(len(gt_traj)), traj_ate, traj_aoe)).T
        # print(traj_s)
        s = np.count_nonzero((traj_s[:, 1] <= epsilo) & (traj_s[:, 2] <= phi))
        return s / len(traj_s)


class TartanAirEvaluator:
    def __init__(self, scale=False, round=1):
        self.ate_eval = ATEEvaluator()
        self.rpe_eval = RPEEvaluator()
        self.kitti_eval = KittiEvaluator()
        self.sr_eval = SREvaluator()

    def evaluate_one_trajectory(
        self, gt_traj_name, est_traj_name, scale=False, sr_t: float = 1.5, sr_r: float = 0.5, delta: int = 5
    ):
        """
        scale = True: calculate a global scale
        """
        # load trajectories
        gt_traj = np.loadtxt(gt_traj_name)
        est_traj = np.loadtxt(est_traj_name)

        if gt_traj.shape[0] != est_traj.shape[0]:
            raise Exception("POSEFILE_LENGTH_ILLEGAL")
        if gt_traj.shape[1] != 7 or est_traj.shape[1] != 7:
            raise Exception("POSEFILE_FORMAT_ILLEGAL")

        # transform and scale
        gt_traj_trans, est_traj_trans, s = robT.transform_trajs(gt_traj, est_traj, scale)
        gt_SEs, est_SEs = robT.quats2SEs(gt_traj_trans, est_traj_trans)

        ate_score, gt_ate_aligned, est_ate_aligned = self.ate_eval.evaluate(gt_traj, est_traj, scale)
        rpe_score = self.rpe_eval.evaluate(gt_SEs, est_SEs)
        kitti_score = self.kitti_eval.evaluate(gt_SEs, est_SEs)
        sr = self.sr_eval.evaluate(gt_SEs, est_SEs, gt_traj, est_traj, scale=scale, epsilo=sr_t, phi=sr_r, delta=delta)

        print("sr_t:", sr_t, " sr_r: ", sr_r, "delta:", delta)

        return {
            "ate_score": ate_score,
            "success rate": sr,
            "rpe_score(r,t)": rpe_score,
            "kitti_score(r,t)": kitti_score,
        }

    def evaluate_one_trajectory_np(
        self, gt_traj, est_traj, scale=False, sr_t: float = 1.5, sr_r: float = 0.5, delta: int = 5
    ):
        """
        scale = True: calculate a global scale
        """
        if gt_traj.shape[0] != est_traj.shape[0]:
            raise Exception("POSEFILE_LENGTH_ILLEGAL")
        if gt_traj.shape[1] != 7 or est_traj.shape[1] != 7:
            raise Exception("POSEFILE_FORMAT_ILLEGAL")

        # transform and scale
        gt_traj_trans, est_traj_trans, s = robT.transform_trajs(gt_traj, est_traj, scale)
        gt_SEs, est_SEs = robT.quats2SEs(gt_traj_trans, est_traj_trans)

        ate_score, gt_ate_aligned, est_ate_aligned = self.ate_eval.evaluate(gt_traj, est_traj, scale)
        rpe_score = self.rpe_eval.evaluate(gt_SEs, est_SEs)
        kitti_score = self.kitti_eval.evaluate(gt_SEs, est_SEs)
        sr = self.sr_eval.evaluate(gt_SEs, est_SEs, gt_traj, est_traj, scale=scale, epsilo=sr_t, phi=sr_r, delta=delta)

        print("sr_t:", sr_t, " sr_r: ", sr_r, "delta:", delta)

        return {
            "ate_score": ate_score,
            "success rate": sr,
            "rpe_score(r,t)": rpe_score,
            "kitti_score(r,t)": kitti_score,
        }


if __name__ == "__main__":

    # scale = True for monocular track, scale = False for stereo track
    aicrowd_evaluator = TartanAirEvaluator()
    result = aicrowd_evaluator.evaluate_one_trajectory("pose_gt.txt", "pose_est.txt", scale=True)
    print(result)
