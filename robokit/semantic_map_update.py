#!/usr/bin/env python

"""Test GroundingSAM on ros images (cv2 & cv_bridge version)"""

import threading
import numpy as np
import rospy
import cv2
from PIL import Image as PILImg
from cv_bridge import CvBridge, CvBridgeError

import networkx as nx
from networkx import Graph
from sensor_msgs.msg import Image
from robokit.perception import GroundingDINOObjectPredictor, SegmentAnythingPredictor
from robokit.utils import annotate, overlay_masks, combine_masks, filter_large_boxes, filter
from shapely.geometry import Point, Polygon
from std_msgs.msg import Int32
from visualization_msgs.msg import Marker, MarkerArray

from listener import ImageListener
import time
from utils import (
    pose_in_map_frame,
    is_nearby_in_map,
    read_graph_json,
    save_graph_json,
    get_fov_points_in_map
)

lock = threading.Lock()

class robokitRealtime:
    def __init__(self):
        # ROS 노드 초기화
        rospy.init_node("seg_rgb")

        self.listener = ImageListener(camera="Fetch")
        self.bridge = CvBridge()

        self.counter = 0
        self.output_dir = "output/real_world"

        # 네트워크 초기화
        self.text_prompt = "table . door . chair ."
        self.gdino = GroundingDINOObjectPredictor()
        self.SAM = SegmentAnythingPredictor()
        self.threshold = {"table": 2, "chair": 0.6, "door": 2}

        self.image_pub = rospy.Publisher("seg_image", Image, queue_size=10)
        self.graph = read_graph_json("graph.json")
        self.pose_list = {"table": [], "chair": [], "door": []}
        self.pause = 0
        rospy.Subscriber("/yes_no", Int32, self.pause_callback)
        time.sleep(5)
        self.marker_pub = rospy.Publisher("graph_nodes", MarkerArray, queue_size=10)

    def pause_callback(self, data):
        self.pause = data.data

    def create_marker(self, pose, category, node_id):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = category
        marker.id = node_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = pose[0]
        marker.pose.position.y = pose[1]
        marker.pose.position.z = 0
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.3
        marker.color.a = 1.0

        if category == "table":
            marker.color.r, marker.color.g, marker.color.b = 0.0, 0.0, 1.0
        elif category == "chair":
            marker.color.r, marker.color.g, marker.color.b = 0.0, 1.0, 0.0
        elif category == "door":
            marker.color.r, marker.color.g, marker.color.b = 1.0, 0.0, 0.0

        return marker

    def publish_graph_to_rviz(self):
        marker_array = MarkerArray()
        node_id = 0
        for node, data in self.graph.nodes(data=True):
            marker = self.create_marker(data["pose"], data["category"], node_id)
            marker_array.markers.append(marker)
            node_id += 1
        self.marker_pub.publish(marker_array)

    def run_network(self):
        iter_ = 0
        while not rospy.is_shutdown():
            with lock:
                if self.listener.im is None:
                    continue
                im_color = self.listener.im.copy()
                depth_img = self.listener.depth.copy()
                rgb_frame_id = self.listener.rgb_frame_id
                rgb_frame_stamp = self.listener.rgb_frame_stamp
                RT_camera, RT_base = self.listener.RT_camera, self.listener.RT_base

            print("===========================================")
            im = cv2.cvtColor(im_color, cv2.COLOR_BGR2RGB)
            img_pil = PILImg.fromarray(im)

            bboxes, phrases, gdino_conf = self.gdino.predict(img_pil, self.text_prompt, 0.55, 0.55)
            bboxes, gdino_conf, phrases, flag = filter(bboxes, gdino_conf, phrases, 1, 0.8, 0.8, 0.8, 0.01, True)

            if flag:
                fov_points = get_fov_points_in_map(depth_img, RT_camera, RT_base)
                fov = Polygon(fov_points)
                nodes_in_fov = {node: data["category"] for node, data in self.graph.nodes(data=True)
                                if fov.contains(Point(data["pose"]))}
                for node in nodes_in_fov:
                    print(f"node is being removed {node}")
                    self.graph.remove_node(node)
                continue

            if len(phrases) == 0:
                print(f"skipping zero phrases \n")
                continue

            w, h = im.shape[1], im.shape[0]
            image_pil_bboxes = self.gdino.bbox_to_scaled_xyxy(bboxes, w, h)
            image_pil_bboxes, masks = self.SAM.predict(img_pil, image_pil_bboxes)
            image_pil_bboxes, index = filter_large_boxes(image_pil_bboxes, w, h, threshold=0.5)
            masks = masks[index]

            detected_poses = {"door": [], "chair": [], "table": []}
            for i, mask in enumerate(masks.cpu().numpy()):
                pose = pose_in_map_frame(RT_camera, RT_base, depth_img, segment=mask[0])
                if pose is not None:
                    detected_poses[phrases[i]].append(pose)

            phrase_iter_ = {"table": 0, "door": 0, "chair": 0}
            for category in detected_poses:
                for pose in detected_poses[category]:
                    _, is_nearby = is_nearby_in_map(self.pose_list[category], pose, threshold=self.threshold[category])
                    if not is_nearby:
                        node_id = f"new_{category}_{iter_}_{phrase_iter_[category]}"
                        self.graph.add_node(node_id, id=node_id, pose=pose, robot_pose=RT_base.tolist(), category=category)
                        phrase_iter_[category] += 1
                    self.pose_list[category].append(pose)

            mask_combined = combine_masks(masks[:, 0, :, :]).cpu().numpy()
            gdino_conf = gdino_conf[index]
            phrases = [phrases[i] for i in np.where(index)[0]]
            bbox_annotated_pil = annotate(overlay_masks(img_pil, masks), image_pil_bboxes, gdino_conf, phrases)
            im_label = np.array(bbox_annotated_pil)

            try:
                rgb_msg = self.bridge.cv2_to_imgmsg(im_label, encoding="rgb8")
                rgb_msg.header.stamp = rgb_frame_stamp
                rgb_msg.header.frame_id = rgb_frame_id
                self.image_pub.publish(rgb_msg)
            except CvBridgeError as e:
                rospy.logerr(f"CVBridge Error: {str(e)}")

            self.publish_graph_to_rviz()
            iter_ += 1


if __name__ == "__main__":
    robokit_instance = robokitRealtime()
    robokit_instance.run_network()
    print(f"closing script! saving graph")
    save_graph_json(robokit_instance.graph, file="graph_updated.json")

