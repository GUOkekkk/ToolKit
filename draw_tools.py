import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import utils.robot_transformation as robT


# define a function to draw the trajectory(position and rotation respectively)
def traj_draw_two(groundtruth, res, fig_dir=None, is_show: bool = False, is_save: bool = True):
    # Draw the result to check
    fig1 = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter(
        groundtruth[0, 0], groundtruth[0, 1], groundtruth[0, 2], color="r", marker="o", s=150, label="start point"
    )
    ax.scatter(groundtruth[:, 0], groundtruth[:, 1], groundtruth[:, 2], c="b", label="Ground truth", s=40)
    ax.scatter(res[:, 0], res[:, 1], res[:, 2], c="green", marker="x", label="Result of PnP", s=40)

    ax.set_title("The position of the camera on the world frame")
    ax.set_zlabel("z", fontdict={"size": 10, "color": "blue"})
    ax.set_ylabel("y", fontdict={"size": 10, "color": "green"})
    ax.set_xlabel("x", fontdict={"size": 10, "color": "red"})
    plt.legend()

    if is_save:
        plt.savefig(fig_dir + "_trajectory.png")
        print("The trajectory plot is stored in", fig_dir + "_trajectory.png")

    if is_show:
        plt.show()

    fig2 = plt.figure()
    plt.subplot(4, 1, 1)
    plt.title("Rotation of x")
    plt.plot(range(len(groundtruth)), groundtruth[:, 3], c="b", label="Ground truth")
    plt.plot(range(len(groundtruth)), res[:, 3], c="r", label="Result of PnP")
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.title("Rotation of y")
    plt.plot(range(len(groundtruth)), groundtruth[:, 4], c="b", label="Ground truth")
    plt.plot(range(len(groundtruth)), res[:, 4], c="r", label="Result of PnP")
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.title("Rotation of z")
    plt.plot(range(len(groundtruth)), groundtruth[:, 5], c="b", label="Ground truth")
    plt.plot(range(len(groundtruth)), res[:, 5], c="r", label="Result of PnP")
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.title("w")
    plt.plot(range(len(groundtruth)), groundtruth[:, 6], c="b", label="Ground truth")
    plt.plot(range(len(groundtruth)), res[:, 6], c="r", label="Result of PnP")
    plt.legend()
    plt.suptitle("Rotation in quaternion")

    if is_save:
        plt.savefig(fig_dir + "_rot.png")
        print("The rotation in quaternion is stored in", fig_dir + "_rot.png")

    if is_show:
        plt.show()

    return 0


def get_camera_wireframe(scale: float = 0.5):  # pragma: no cover
    """
    Returns a wireframe of a 3D line-plot of a camera symbol.
    """
    a = 0.5 * np.asarray([-2, 1.5, 4])
    up1 = 0.5 * np.asarray([0, 1.5, 4])
    up2 = 0.5 * np.asarray([0, 2, 4])
    b = 0.5 * np.asarray([2, 1.5, 4])
    c = 0.5 * np.asarray([-2, -1.5, 4])
    d = 0.5 * np.asarray([2, -1.5, 4])
    C = np.zeros(3)
    F = np.asarray([0, 0, 3])
    camera_points = [a, up1, up2, up1, b, d, c, a, C, b, d, C, c, C, F]
    lines = np.stack([x for x in camera_points]) * scale
    return lines


def plot_camera(ax, camera, color: str = "blue", scale: float = 0.5):
    """
    Plots a set of `cameras` objects into the maplotlib axis `ax` with
    color `color`.
    """
    cam_wires_canonical = get_camera_wireframe(scale=scale)
    cam_trans = robT.get_T_from_pose(camera_pose=camera)
    cam_wires_trans = robT.transform_points(cam_trans, cam_wires_canonical)
    plot_handles = []
    # the Z and Y axes are flipped intentionally here!
    x_, y_, z_ = cam_wires_trans[:, 0], cam_wires_trans[:, 1], cam_wires_trans[:, 2]
    # x_, y_, z_ = cam_wires_canonical[:, 0], cam_wires_canonical[:, 1], cam_wires_canonical[:, 2]
    (h,) = ax.plot(x_, y_, z_, color=color, linewidth=1)
    plot_handles.append(h)
    return plot_handles


# define a function to draw the trajectory(position and rotation together)
def traj_draw(groundtruth, res, fig_dir=None, is_show: bool = False, is_save: bool = True, scale: float = 0.5):
    # Draw the result to check
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.clear()
    ax.set_title("The trajectory")
    for i in range(len(groundtruth)):
        handle_cam = plot_camera(ax, res[i], color="#FF7D1E", scale=scale)
        handle_cam_gt = plot_camera(ax, groundtruth[i], color="#812CE5", scale=scale)

    # plot_radius = 15
    # ax.set_xlim3d([-plot_radius, plot_radius])
    # ax.set_ylim3d([3 - plot_radius, 3 + plot_radius])
    # ax.set_zlim3d([-plot_radius, plot_radius])
    ax.set_zlabel("z", fontdict={"size": 10, "color": "blue"})
    ax.set_ylabel("y", fontdict={"size": 10, "color": "green"})
    ax.set_xlabel("x", fontdict={"size": 10, "color": "red"})
    labels_handles = {
        "Estimated cameras": handle_cam[0],
        "GT cameras": handle_cam_gt[0],
    }
    ax.legend(
        labels_handles.values(),
        labels_handles.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
    )
    if is_show:
        plt.show()

    if is_save:
        plt.savefig(fig_dir + "_oren_trajectory.png")
        print("The trajectory plot is stored in", fig_dir + "_oren_trajectory.png")

    return 0
