# This is tool kit to get the pointcloud or visualize the pointcloud based on the Open3d and OpenCV
# Created by Ke GUO

import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import open3d as o3d
import copy
from tqdm import tqdm
import time
import pandas as pd
import robot_transformation as robT


# define a small function to draw the depth and color ilg together by using the plt
def draw_depth_color(depthimg: np.ndarray, colorimg: np.ndarray):

    """
    plot the depth and color image together

    Args:
      _depthimg (np.ndarray): the depth image
      _colorimg (np.ndarray): the color image

    Returns:
      plot

    """
    print("Check the depth and color image", "\n")
    plt.figure(figsize=(15, 20))

    plt.subplot(1, 2, 1)
    plt.imshow(depthimg, cmap=plt.cm.gray)  # set to the gray plot
    plt.title("Depth Image")

    plt.subplot(1, 2, 2)
    _imgRGB = cv2.cvtColor(colorimg.astype("uint8"), cv2.COLOR_BGR2RGB)  # change the color channel
    plt.imshow(_imgRGB.astype(np.float64) / 255.0)
    plt.title("Color Image")

    plt.show()

    return 0


# define a function to get the pointcloud from the depth image
def get_pointcloud_from_depth(
    depthimg: np.ndarray, camera_paras: np.ndarray, flatten: bool = True, threshold: int = 10000
) -> np.ndarray:
    """Create point clouds from the depth image

    Args:
        _depthimg (np.ndarray): the depth image
        _camera_paras (np.ndarray): the paras of the camera, in the order (fx, fy, cx, cy)

    Returns:
        _pointclouds (np.ndarray): pointclouds <float: num_points, 3>
    """
    _fx, _fy, _cx, _cy = camera_paras

    # print(np.max(_depthimg))
    _depthimg = np.where(depthimg > threshold, threshold, depthimg)
    # print(np.max(_depthimg))

    _h, _w = np.mgrid[0 : _depthimg.shape[0], 0 : _depthimg.shape[1]]
    _z = _depthimg
    _x = (_w - _cx) * _z / _fx
    _y = (_h - _cy) * _z / _fy

    pointclouds = np.dstack((_x, _y, _z)) if flatten is False else np.dstack((_x, _y, _z)).reshape(-1, 3)

    return pointclouds


# define a function to visualize the pointcloud without the color
def draw_pointclouds_inblack(
    _pointclouds: np.ndarray,
    _window_name: str = "Pointcloud",
    _background_color: np.ndarray = np.asarray([0, 0, 0]),
    _pointcloud_color: np.ndarray = np.asarray([1, 1, 1]),
    _camera_pose: np.ndarray = np.asarray([0, 0, 0, 0, 0, 0, 0]),
    _frame_size: int = 1,
    _frame_origin: list = [0, 0, 0],
    _point_size: int = 1,
):
    """
    To visualize the pointcloud, maybe only for Ke, cause we should use the open3d

    Args:
        _pointclouds (np.ndarray):
        _window_name (str): Defaults to 'Pointcloud'.
        _background_color (np.ndarray, optional): Defaults to np.asarray([0, 0, 0]).
        _camera_pose (np.ndarray, optional): Defaults to np.asarray([0, 0, 0, 0, 0, 0, 0]). The camera pose in the
        world frame <1, 7> rotaiion vector in quaternion form.
        _pointcloud_color (np.ndarray, optional): Defaults to np.asarray([1, 1, 1]).
        _frame_size (int, optional):  Defaults to 1.
        _frame_origin (list, optional): Defaults to [0, 0, 0].
        _point_size (int): Defaults to 1.

    Returns:
        _type_: _description_
    """

    print("Visualize the pointcloud in white:", "\n")
    _vis = o3d.visualization.Visualizer()
    _vis.create_window(window_name=_window_name)

    _vis.get_render_option().background_color = _background_color  # set the color of the background

    _pointclouds_o3d = o3d.geometry.PointCloud()
    _pointclouds_o3d.points = o3d.utility.Vector3dVector(_pointclouds)

    _pointclouds_o3d.paint_uniform_color(_pointcloud_color)  # set the color of the pointclouds
    _vis.get_render_option().point_size = _point_size

    _mesh_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=_frame_size, origin=_frame_origin)

    _mesh_camera = copy.deepcopy(_mesh_origin).transform(
        robT.get_T_from_pose(_camera_pose)
    )  # Maybe we could use other method instead of the copy

    _vis.add_geometry(_pointclouds_o3d)
    _vis.add_geometry(_mesh_origin)
    _vis.add_geometry(_mesh_camera)

    _vis_controller = _vis.get_view_control()
    _vis_controller.set_front((0, 0, -1))
    _vis_controller.set_lookat((1, 0, 0))
    _vis_controller.set_up((0, -1, 0))

    _vis.run()

    _vis.destroy_window()  # The destroy_window() is necessary otherwise the pose can not update

    return 0


# define a function to visualize the pointcloud with the color
def draw_pointclouds_incolor(
    _pointclouds: np.ndarray,
    _pointclouds_color: np.ndarray,
    _camera_pose: np.ndarray = np.asarray([0, 0, 0, 0, 0, 0, 1]),
    _window_name: str = "Pointcloud",
    _frame_size: int = 1,
    _frame_origin: list = [0, 0, 0],
    _point_size: int = 1,
):
    """To visualize the pointcloud, maybe only for Ke, cause we should use the open3d

    Args:
        _pointclouds (np.ndarray):
        _window_name (str): Defaults to'Pointcloud'
        _frame_size (int, optional):  Defaults to 1.
        _camera_pose (np.ndarray, optional): Defaults to np.asarray([0, 0, 0, 0, 0, 0, 0]). The camera pose in the
        world frame <nums, 7> rotaiion vector in quaternion form.
        _frame_origin (list, optional): Defaults to [0, 0, 0].
        _point_size (int): Defaults to 1.

    Returns:
        _type_: _description_
    """
    print("Visualize the pointcloud in color:", "\n")
    _vis = o3d.visualization.Visualizer()
    _vis.create_window(window_name=_window_name)

    _pointclouds_o3d = o3d.geometry.PointCloud()

    # _vis.get_render_option().background_color = [0, 0, 0] # set the backgroud to black
    # Set a threshold to filter some large depth pointclouds
    # ? Does not work, idk why but maybe modify the depth image directly is better
    # _threshold = 0.5
    # print(np.max(_pointclouds[:, 1]))
    # _pointclouds[:, 1] = np.where(_pointclouds[:, 1] < _threshold, _pointclouds[:, 1], _threshold)
    # print(np.max(_pointclouds[:, 1]))

    _pointclouds_o3d.points = o3d.utility.Vector3dVector(_pointclouds)

    _pointclouds_color_RGB = cv2.cvtColor(_pointclouds_color.astype("uint8"), cv2.COLOR_BGR2RGB)
    _pointclouds_color_RGB = (
        _pointclouds_color_RGB.reshape(-1, 3).astype(np.float64) / 255.0
    )  # the o3d is different as plt, the color should go to [0, 1]
    _pointclouds_o3d.colors = o3d.utility.Vector3dVector(_pointclouds_color_RGB)

    _vis.get_render_option().point_size = _point_size

    _mesh_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=_frame_size, origin=_frame_origin)

    _mesh_camera = copy.deepcopy(_mesh_origin).transform(
        robT.get_T_from_pose(_camera_pose)
    )  # Maybe we could use other method instead of the copy

    _vis.add_geometry(_pointclouds_o3d)
    _vis.add_geometry(_mesh_origin)
    _vis.add_geometry(_mesh_camera)

    _vis_controller = _vis.get_view_control()  # choose a BEV angle
    _vis_controller.set_front((0, -1, 0))
    _vis_controller.set_lookat((0, 0, 1))
    _vis_controller.set_up((-1, 0, 0))

    _vis.run()

    _vis.destroy_window()  # The destroy_window() is necessary otherwise the pose can not update

    return 0


# define a function to correct the pointcloud(remove the rotation of x and z) based on the camera pose
def correct_pc_rotationxz(pointcloud: np.ndarray, camera_pose: np.ndarray = np.asarray([0, 0, 0, 0, 0, 0, 1])):
    """
    Correct the pointcloud(remove the rotation of x and z) based on the camera pose

    Args:
        _pointcloud (np.ndarray): <nums, 3>
        _camera_pose (np.ndarray, optional): The camera pose in the world frame <nums, 7> rotaiion vector in
        quaternion form. Defaults to np.asarray([0, 0, 0, 0, 0, 0, 1]).
                                            Note: it should be the camera pose relative to the the pose 0

    Returns:
        _pointcloud: <nums, 3>
    """
    _r = R.from_quat(camera_pose[3:]).as_euler("xyz")
    _r[1] = 0  # keep the rotation along y
    _R_mat = R.from_euler("xyz", _r).as_matrix()

    pointcloud = _R_mat.T.dot(pointcloud.T).T

    return pointcloud


# define a function to create the BEV image from the pointcloud with color
def get_BEV_from_pointcloud(pointcloud: np.ndarray, BEV_size: np.ndarray, pc_shape: np.ndarray):
    """
      Create the BEV image from the pointcloud with color
    Args:
        _pointcloud (np.ndarray): <nums, 6>
        _size (np.ndarray): _description_

    Returns:
        _type_: _description_
    """
    start = time.time()

    _BEV_image = np.repeat(255 * np.ones(BEV_size)[:, :, None], 3, axis=-1)  # Maybe the background in white is better

    _Xindex = pointcloud[:, 0] // pc_shape[0]  # Becasue here, some information are lost
    _Zindex = pointcloud[:, 2] // pc_shape[1]

    _df = pd.DataFrame({"Xindex": _Xindex, "Y": pointcloud[:, 1], "Zindex": _Zindex})
    _df = (
        _df.groupby(["Xindex", "Zindex"])
        .idxmin()
        .reset_index()  # test the performance with the min method, min should be correct
        # _df.groupby(["Xindex", "Zindex"]).idxmax().reset_index()
    )  # use idxmax() or idxmin() to change to the opposite view

    _index = np.array(_df).astype(np.int32)

    _BEV_image[_index[:, 0], _index[:, 1]] = pointcloud[_index[:, 2], 3:]

    BEV_image_RGB = cv2.cvtColor(_BEV_image.astype("uint8"), cv2.COLOR_BGR2RGB)

    print("In", time.time() - start, "seconds get the BEV")

    # plt.imshow(_BEV_image_RGB.astype(np.float64) / 255.0)
    # plt.show()

    return BEV_image_RGB


# define a function to create the BEV from the depth and color image
def get_BEV_from_depthandcolor(
    depth: np.ndarray,
    img: np.ndarray,
    camera_pose: np.ndarray,
    camera_paras: np.ndarray,
    BEV_size: np.ndarray = np.asarray((240, 320)),
    threshold: int = 10000,
    n_slice: int = 3,
    pc_range: np.ndarray = np.asarray((15, 20)),
    pc_shift: float = 10.0,
):
    """
    Create the BEV from the pointcloud with the color
    Args:
        depth (np.ndarray): <480, 640, 1>
        img (np.ndarray): <480, 640, 3>
        camera_pose (np.ndarray): The camera pose in the world frame <nums, 7> rotaiion vector in quaternion form.
                                           Note: it should be the camera pose relative to the the pose 0
        camera_paras (np.ndarray): the paras of the camera, in the order (fx, fy, cx, cy)
        size (tuple): <w, h>
        threshold (int):
        n_slice (int):

    Returns:
        BEV_image: <n_slice, w, h, 3>
    """

    _pointcloud = get_pointcloud_from_depth(
        depth, camera_paras, threshold=threshold
    )
    # The frame is based on the camera frame! So we actually take the x and z, not z. We should know we are
    # in which frmae everytime.
    _pointcloud = correct_pc_rotationxz(_pointcloud, camera_pose)

    # print(_pointcloud)

    _pointcloud = np.hstack((_pointcloud, img.reshape(-1, 3)))  # <nums, 6>

    # _min = np.min(_pointcloud[:, [0, 2]], axis=0) - 1e-8
    # _max = np.max(_pointcloud[:, [0, 2]], axis=0) + 1e-8

    # print('min is', _min, 'max is', _max)
    _pointcloud[:, 0] = _pointcloud[:, 0] + pc_shift  # make sure all points are positive for the BEV img

    # _pointcloud = _pointcloud[((0 < _pointcloud[:, 0] < _range[0]) & (0 < _pointcloud[:, 2] < _range[1]))]
    _pointcloud = _pointcloud[
        ((_pointcloud[:, 0] < pc_range[0]) & (_pointcloud[:, 0] > 0) & (_pointcloud[:, 2] < pc_range[1]))
    ]

    print("Pointcloud shape is ", _pointcloud.shape)

    _shape = pc_range / BEV_size
    # print("shape is", _shape)

    # print('shape is', _shape)
    # print(np.max(_pointcloud[:, 1])) # It seems that we do not need the sort() here, the pointcloud is already
    # sorted, but for the extra safe, i did the sort again
    _pointcloud = _pointcloud[np.argsort(_pointcloud[:, 1])]

    # Check if the Pointcloud could be split equaly
    _pointcloud_list = np.array_split(_pointcloud, n_slice, axis=0)

    BEV_image_RGB = np.zeros((n_slice, BEV_size[0], BEV_size[1], 3))

    for i in tqdm(range(n_slice)):
        # print(_pointcloud_list[i].shape)
        BEV_image_RGB[i] = get_BEV_from_pointcloud(_pointcloud_list[i], BEV_size, _shape)

    return BEV_image_RGB


# define a function to change the RGBBEV to a 2D pointcloud
def BEV_to_2dpc(BEV_image: np.ndarray, pc_shape: np.ndarray, pc_shift: float):
    """
    define a function to change the RGBBEV to a 2D pointcloud

    Args:
        _BEV_image (np.ndarray): _description_

    Returns:
        pc_2d: _description_
    """
    pc_2d = np.where(BEV_image < 254.9999, 255.0, 0.0)
    pc_2d = pc_2d[:, :, 0]

    # print("SHAPE is ", _shape)

    pc_2d = np.asarray(np.where(pc_2d[:, :] == 255.0)).astype(np.float64).T

    pc_2d[:, 0] = pc_2d[:, 0] * pc_shape[0]
    pc_2d[:, 1] = pc_2d[:, 1] * pc_shape[1]
    pc_2d[:, 0] = pc_2d[:, 0] - pc_shift

    return pc_2d


# define a funcion to get two 2d pointclouds from two 2d BEV, matched by using ORB feature
def ORB_BEV_to_2dpc(BEV_0: np.ndarray, BEV_1: np.ndarray, pc_shape: np.ndarray, pc_shift: float):

    # Extract the orb
    _orb = cv2.ORB_create()
    print(BEV_0.shape)
    _kp0, _des0 = _orb.detectAndCompute(BEV_0.astype(np.uint8), None)
    _kp1, _des1 = _orb.detectAndCompute(BEV_1.astype(np.uint8), None)

    # match kpts
    _bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    _matches = _bf.match(_des0, _des1)

    _min_distance = _matches[0].distance

    for _x in _matches:
        if _x.distance < _min_distance:
            _min_distance = _x.distance

    _good_match = []
    for _x in _matches:
        if _x.distance <= max(2 * _min_distance, 30):
            _good_match.append(_x)

    # organize key points into matrix, each row is a point
    # The return of the .pt is the x and y of the feature
    # pc_2D_0 = np.array([_kp0[m.queryIdx].pt for m in _good_match]).reshape((-1, 2))  # shape: <num_pts, 2>
    # pc_2D_1 = np.array([_kp1[m.trainIdx].pt for m in _good_match]).reshape((-1, 2))  # shape: <num_pts, 2>

    pc_2D_0 = np.array([_kp0[m.queryIdx].pt for m in _matches]).reshape((-1, 2))  # shape: <num_pts, 2>
    pc_2D_1 = np.array([_kp1[m.trainIdx].pt for m in _matches]).reshape((-1, 2))  # shape: <num_pts, 2>

    pc_2D_0[:, 0] = pc_2D_0[:, 0] * pc_shape[0]
    pc_2D_0[:, 1] = pc_2D_0[:, 1] * pc_shape[1]
    pc_2D_0[:, 0] = pc_2D_0[:, 0] - pc_shift

    pc_2D_1[:, 0] = pc_2D_1[:, 0] * pc_shape[0]
    pc_2D_1[:, 1] = pc_2D_1[:, 1] * pc_shape[1]
    pc_2D_1[:, 0] = pc_2D_1[:, 0] - pc_shift

    # res = cv2.drawMatches(image_0[i].astype(np.uint8), _kp0, image_1[i].astype(np.uint8), _kp1, matches, outImg=None)
    # cv2.namedWindow("Match Result", 0)
    # cv2.resizeWindow("Match Result", 1000, 1000)
    # cv2.imshow("Match Result", res)
    # cv2.waitKey(0)

    return pc_2D_0, pc_2D_1


# a test function
def test_draw_pointclouds(
    _pointclouds: np.ndarray,
    _pointclouds_ry: np.ndarray,
    _pointclouds_color: np.ndarray,
    _camera_pose: np.ndarray = np.asarray([0, 0, 0, 0, 0, 0, 1]),
    _window_name: str = "Pointcloud",
    _frame_size: int = 1,
    _frame_origin: list = [0, 0, 0],
    _point_size: int = 1,
):
    """To visualize the pointcloud, maybe only for Ke, cause we should use the open3d

    Args:
        _pointclouds (np.ndarray):
        _window_name (str): Defaults to'Pointcloud'
        _frame_size (int, optional):  Defaults to 1.
        _camera_pose (np.ndarray, optional): Defaults to np.asarray([0, 0, 0, 0, 0, 0, 1]).
        The camera pose in the world frame <nums, 7> rotaiion vector in quaternion form.
        _frame_origin (list, optional): Defaults to [0, 0, 0].
        _point_size (int): Defaults to 1.

    Returns:
        _type_: _description_
    """
    print("Visualize the pointcloud in color:", "\n")
    _vis = o3d.visualization.Visualizer()
    _vis.create_window(window_name=_window_name)

    _pointclouds_o3d = o3d.geometry.PointCloud()

    # _vis.get_render_option().background_color = [0, 0, 0] # set the backgroud to black

    _pointclouds_o3d.points = o3d.utility.Vector3dVector(_pointclouds)

    _pointclouds_color_RGB = cv2.cvtColor(_pointclouds_color.astype("uint8"), cv2.COLOR_BGR2RGB)
    _pointclouds_color_RGB = (
        _pointclouds_color_RGB.reshape(-1, 3).astype(np.float64) / 255.0
    )  # the o3d is different as plt, the color should go to [0, 1]
    _pointclouds_o3d.colors = o3d.utility.Vector3dVector(_pointclouds_color_RGB)

    _vis.get_render_option().point_size = _point_size

    _mesh_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=_frame_size, origin=_frame_origin)

    _mesh_camera = copy.deepcopy(_mesh_origin).transform(
        robT.get_T_from_pose(_camera_pose)
    )  # Maybe we could use other method instead of the copy

    # For the test
    _pointclouds_o3d_ry = o3d.geometry.PointCloud()

    # _vis.get_render_option().background_color = [0, 0, 0] # set the backgroud to black
    _pointclouds_o3d_ry.points = o3d.utility.Vector3dVector(_pointclouds_ry)

    _vis.add_geometry(_pointclouds_o3d)
    _vis.add_geometry(_pointclouds_o3d_ry)
    _vis.add_geometry(_mesh_origin)
    _vis.add_geometry(_mesh_camera)

    _vis_controller = _vis.get_view_control()
    _vis_controller.set_front((0, 0, -1))
    _vis_controller.set_lookat((1, 0, 0))
    _vis_controller.set_up((0, -1, 0))

    _vis.run()

    _vis.destroy_window()  # The destroy_window() is necessary otherwise the pose can not update

    return 0


# a test function
def get_frontview_from_depthandcolor(
    _depth: np.ndarray,
    _img: np.ndarray,
    _camera_pose: np.ndarray,
    _camera_paras: np.ndarray,
    _size: np.ndarray = (240, 320),
    _threshold: int = 10000,
    _N_slice: int = 3,
):
    """
    Create the BEV from the pointcloud with the color
    Args:
        _depth (np.ndarray): <480, 640, 1>
        _img (np.ndarray): <480, 640, 3>
        _camera_pose (np.ndarray): The camera pose in the world frame <nums, 7> rotaiion vector in quaternion form.
                                            Note: it should be the camera pose relative to the the pose 0
        _camera_paras (np.ndarray): the paras of the camera, in the order (fx, fy, cx, cy)
        _size (tuple): _description_
        _threshold (int):
        _N_slice (int):

    Returns:
        _BEV_image: <size>
    """
    start = time.time()

    _pointcloud = get_pointcloud_from_depth(
        _depth, _camera_paras, _threshold=_threshold
    )
    # The frame is based on the camera frame! So we actually take the x and z, not z.
    # We should know we are in which frmae everytime.
    _pointcloud = correct_pc_rotationxz(_pointcloud, _camera_pose)

    _pointcloud = np.hstack((_pointcloud, _img.reshape(-1, 3)))  # <nums, 6>

    _min = np.min(_pointcloud[:, [1, 0]], axis=0) - 1e-8
    _max = np.max(_pointcloud[:, [1, 0]], axis=0) + 1e-8

    print("min is", _min, "max is", _max)

    _shape = (_max - _min) / _size

    print("shape is", _shape)

    _BEV_image = np.repeat(255 * np.ones(_size)[:, :, None], 3, axis=-1)  # Maybe the background in white is better

    _pointcloud[:, [1, 0]] = _pointcloud[:, [1, 0]] - _min
    _Xindex = _pointcloud[:, 0] // _shape[1]
    _Yindex = _pointcloud[:, 1] // _shape[0]

    _df = pd.DataFrame({"Xindex": _Xindex, "Yindex": _Yindex, "Zindex": _pointcloud[:, 2]})  # ?
    print(_df)
    _df = _df.groupby(["Yindex", "Xindex"]).idxmin().reset_index()  # ?
    print(_df)

    _index = np.array(_df).astype(np.int32)

    print(_index.shape)

    print(_index[:, 2])

    _BEV_image[_index[:, 0], _index[:, 1]] = _pointcloud[_index[:, 2], 3:]  # ?
    _BEV_image_RGB = cv2.cvtColor(_BEV_image.astype("uint8"), cv2.COLOR_BGR2RGB)

    print("In", time.time() - start, "seconds get the BEV")

    plt.imshow(_BEV_image_RGB.astype(np.float64) / 255.0)
    plt.show()

    return 0


# a test function
def get_sideview_from_depthandcolor(
    _depth: np.ndarray,
    _img: np.ndarray,
    _camera_pose: np.ndarray,
    _camera_paras: np.ndarray,
    _size: np.ndarray = (240, 320),
    _threshold: int = 10000,
    _N_slice: int = 3,
):
    """
    Create the BEV from the pointcloud with the color
    Args:
        _depth (np.ndarray): <480, 640, 1>
        _img (np.ndarray): <480, 640, 3>
        _camera_pose (np.ndarray): The camera pose in the world frame <nums, 7> rotaiion vector in quaternion form.
                                            Note: it should be the camera pose relative to the the pose 0
        _camera_paras (np.ndarray): the paras of the camera, in the order (fx, fy, cx, cy)
        _size (tuple): _description_
        _threshold (int):
        _N_slice (int):

    Returns:
        _BEV_image: <size>
    """
    start = time.time()

    _pointcloud = get_pointcloud_from_depth(
        _depth, _camera_paras, _threshold=_threshold
    )
    # The frame is based on the camera frame!
    # So we actually take the x and z, not z. We should know we are in which frmae everytime.
    _pointcloud = correct_pc_rotationxz(_pointcloud, _camera_pose)

    _pointcloud = np.hstack((_pointcloud, _img.reshape(-1, 3)))  # <nums, 6>

    _min = np.min(_pointcloud[:, [1, 2]], axis=0) - 1e-8
    _max = np.max(_pointcloud[:, [1, 2]], axis=0) + 1e-8

    print("min is", _min, "max is", _max)

    _shape = (_max - _min) / _size

    print("shape is", _shape)

    _BEV_image = np.repeat(255 * np.ones(_size)[:, :, None], 3, axis=-1)  # Maybe the background in white is better

    _pointcloud[:, [1, 2]] = _pointcloud[:, [1, 2]] - _min
    _Yindex = _pointcloud[:, 1] // _shape[0]
    _Zindex = _pointcloud[:, 2] // _shape[1]

    _df = pd.DataFrame({"Xindex": _pointcloud[:, 0], "Yindex": _Yindex, "Zindex": _Zindex})  # ?
    print(_df)
    _df = _df.groupby(["Yindex", "Zindex"]).idxmin().reset_index()  # ?
    print(_df)

    _index = np.array(_df).astype(np.int32)

    print(_index.shape)

    print(_index[:, 2])

    _BEV_image[_index[:, 0], _index[:, 1]] = _pointcloud[_index[:, 2], 3:]  # ?
    _BEV_image_RGB = cv2.cvtColor(_BEV_image.astype("uint8"), cv2.COLOR_BGR2RGB)

    print("In", time.time() - start, "seconds get the BEV")

    plt.imshow(_BEV_image_RGB.astype(np.float64) / 255.0)
    plt.show()

    return 0


# define a function to get the pointcloud from the keypoint
def get_pointcloud_from_keypoint(
    keypoints: np.ndarray, depthimg: np.ndarray, camera_paras: np.ndarray, threshold: int = 10000
):
    """_summary_

    Args:
        keypoints (np.ndarray): _description_
        depthimg (np.ndarray): _description_
        camera_paras (np.ndarray): _description_
        flatten (bool, optional): _description_. Defaults to True.
        threshold (int, optional): _description_. Defaults to 10000.

    Returns:
        _type_: _description_
    """
    _fx, _fy, _cx, _cy = camera_paras
    pc3d = np.hstack(
        (
            # ? is this correct?
            keypoints[:, 0].reshape(-1, 1),
            keypoints[:, 1].reshape(-1, 1),
            depthimg[keypoints[:, 1], keypoints[:, 0]].reshape(-1, 1),
        )
    )

    for i in range(len(pc3d)):
        pc3d[i, 0] = (pc3d[i, 0] - _cx) * pc3d[i, 2] / _fx
        pc3d[i, 1] = (pc3d[i, 1] - _cy) * pc3d[i, 2] / _fy
    return pc3d
