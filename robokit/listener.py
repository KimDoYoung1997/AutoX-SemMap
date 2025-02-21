#!/usr/bin/env python
"""
ROS RGB-D Camera and TF Listener Node

This node subscribes to RGB-D camera topics and TF transforms.
It synchronizes RGB and depth images, processes them, and maintains current robot pose information.
The processed data can be used for 3D reconstruction, object detection, or SLAM.

Topics subscribed:
    - /head_camera/rgb/image_raw (sensor_msgs/Image)
    - /head_camera/depth_registered/image_raw (sensor_msgs/Image)
    - /head_camera/rgb/camera_info (sensor_msgs/CameraInfo)
    - tf transforms between relevant frames

Topics published:
    - /lidar_pc (sensor_msgs/PointCloud2) 
"""
import threading
import numpy as np

# from scipy.io import savemat

import rospy
import tf
import tf2_ros
import message_filters
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from cv_bridge import CvBridge

from ros_utils import ros_qt_to_rt

from nav_msgs.msg import Odometry, OccupancyGrid
import time
import yaml
import ros_numpy

lock = threading.Lock()


class ImageListener:
    """
    A class to handle synchronized RGB-D camera data and robot transforms.
    
    This class synchronizes RGB and depth images, processes them into OpenCV format,
    and maintains current transform information between robot frames.
    """
    def __init__(self, camera="Fetch"):
        self.cv_bridge = CvBridge()

        # Initialize image storage variables
        self.im = None                  # RGB image
        self.depth = None               # Depth image
        self.rgb_frame_id = None        # Frame ID of RGB camera
        self.rgb_frame_stamp = None     # Timestamp of RGB frame

        self.base_frame = "base_link"
        self.camera_frame = "head_camera_rgb_optical_frame" # 로봇에 따라 다를 수 있음, Camera's optical frame
        self.target_frame = self.base_frame

        self.tf_listener = tf.TransformListener()
        #  message_filters.Subscriber를 사용하는 주된 이유는 ApproximateTimeSynchronizer와 함께 사용하기 위해서
        rgb_sub = message_filters.Subscriber(
            "/head_camera/rgb/image_raw", Image, queue_size=10
        )
        depth_sub = message_filters.Subscriber(
            "/head_camera/depth_registered/image_raw", Image, queue_size=10
        )
        self.lidar_pub = rospy.Publisher("/lidar_pc", PointCloud2, queue_size=10)
        msg = rospy.wait_for_message("/head_camera/rgb/camera_info", CameraInfo)

        # Get camera calibration parameters
        intrinsics = np.array(msg.K).reshape(3, 3)
        self.intrinsics = intrinsics
        self.fx = intrinsics[0, 0]      # Focal length x
        self.fy = intrinsics[1, 1]      # Focal length y
        self.px = intrinsics[0, 2]      # Principal point x
        self.py = intrinsics[1, 2]      # Principal point y
        print(intrinsics)

        queue_size = 1
        slop_seconds = 0.1
        # RGB 카메라와 깊이 카메라의 데이터를 동기화하여 처리하기 위한 설정입니다
            # ApproximateTimeSynchronizer는 두 센서의 데이터를 시간 기준으로 매칭하고 slop_seconds 내의 시간 차이를 가진 메시지들을 한 쌍으로 묶음
            # 매칭된 메시지 쌍이 생기면 callback_rgbd 함수를 호출
            # 이렇게 함으로써 시간적으로 가까운 RGB와 깊이 데이터만 처리하여 데이터 동기화로 인한 정확한 3D 처리가 가능하고 너무 시간 차이가 나는 데이터는 자동으로 필터링
        ts = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], queue_size, slop_seconds
        )
        ts.registerCallback(self.callback_rgbd)

    def callback_rgbd(self, rgb, depth):
        """
        Callback for synchronized RGB and depth images.
        
        Processes incoming RGB-D data and updates class members with the latest
        images and transform information.

        Args:
            rgb (sensor_msgs/Image): RGB image message
            depth (sensor_msgs/Image): Depth image message
        """
        # get camera pose in base
        try:
            # 로봇의 베이스 기준으로 카메라의 위치(translation)와 방향(rotation)을 얻습니다
            trans, rot = self.tf_listener.lookupTransform(
                self.base_frame, self.camera_frame, rospy.Time(0)
            )
            # ros_qt_to_rt 함수는 ROS의 쿼터니언(quaternion)과 이동(translation) 벡터를 4x4 동차 변환 행렬(homogeneous transformation matrix)로 변환하는 함수
            RT_camera = ros_qt_to_rt(rot, trans)
            # 로봇의 베이스 기준으로 레이저 센서의 위치(translation)와 방향(rotation)을 얻습니다
            self.trans_l, self.rot_l = self.tf_listener.lookupTransform(
                self.base_frame, "laser_link", rospy.Time(0)
            )
            RT_laser = ros_qt_to_rt(self.rot_l, self.trans_l)
            # 전체 지도(map) 기준으로 로봇의 베이스가 어디에 있는지를 구합니다
            self.trans_l, self.rot_l = self.tf_listener.lookupTransform(
                "map", self.base_frame, rospy.Time(0)
            )
            RT_base = ros_qt_to_rt(self.rot_l, self.trans_l)
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn("Update failed... " + str(e))
            RT_camera = None
            RT_laser = None
            RT_base = None

        # depth 이미지 변환
            # ROS의 이미지 메시지를 OpenCV 형식으로 변환해야 이미지 처리 가능
        if depth.encoding == "32FC1":
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth, desired_encoding="32FC1") # 32FC1: 32비트 float 형식
            depth_cv[np.isnan(depth_cv)] = 0
        elif depth.encoding == "16UC1": # Intel RealSense는 주로 16UC1 사용
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth, desired_encoding="16UC1").astype(np.float32) # 16UC1: 16비트 unsigned integer 형식
            depth_cv /= 1000.0          # 단위 통일: mm → m 변환 (16UC1의 경우)
        else:
            rospy.logerr_throttle(
                1,
                "Unsupported depth type. Expected 16UC1 or 32FC1, got {}".format(
                    depth.encoding
                ),
            )
            return

        # RGB 이미지 변환
            # ROS의 이미지 메시지를 OpenCV 형식으로 변환해야 이미지 처리 가능
        im = self.cv_bridge.imgmsg_to_cv2(rgb, desired_encoding="bgr8") # bgr8: 8비트 unsigned integer 형식

        with lock:      # threading.Lock : - 다른 스레드가 데이터를 읽는 동안 데이터가 업데이트되는 것을 방지, Race condition 방지
            self.im = im.copy()          # Store RGB image
            self.depth = depth_cv.copy()  # Store depth image
            self.rgb_frame_id = rgb.header.frame_id
            self.rgb_frame_stamp = rgb.header.stamp
            self.height = depth_cv.shape[0]
            self.width = depth_cv.shape[1]
            self.RT_camera = RT_camera    # Camera to base transform
            self.RT_laser = RT_laser      # Laser to base transform
            self.RT_base = RT_base        # Base to map transform


    def get_data_to_save(self):
        """
        Thread-safe method to retrieve current transform data.

        Returns:
            tuple: (RT_camera, RT_base) or (None, None) if no data available
        """
        with lock:
            if self.im is None:
                return None, None
            RT_camera = self.RT_camera.copy()
            RT_base = self.RT_base.copy()
        return RT_camera, RT_base
if __name__ == "__main__":
    # test_basic_img()
    rospy.init_node("image_listener")
    listener = ImageListener()
    time.sleep(3)
