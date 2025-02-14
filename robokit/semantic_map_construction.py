#!/usr/bin/env python

"""Test GroundingSAM on ros images"""

import threading
import numpy as np
import rospy
from PIL import Image as PILImg

import cv2  # OpenCV 추가
from cv_bridge import CvBridge  # ROS Image 변환을 위한 cv_bridge 추가
import networkx as nx
from networkx import Graph

from sensor_msgs.msg import Image
from robokit.perception import GroundingDINOObjectPredictor, SegmentAnythingPredictor
from robokit.utils import annotate, overlay_masks, combine_masks, filter_large_boxes, filter
from visualization_msgs.msg import Marker, MarkerArray

lock = threading.Lock()
from listener import ImageListener
import time
from utils import (
    pose_in_map_frame,
    is_nearby_in_map,
    save_graph_json
)

class robokitRealtime:

    def __init__(self):
        # initialize a node
        rospy.init_node("seg_rgb")

        self.bridge = CvBridge()  # cv_bridge 객체 생성
        self.listener = ImageListener(camera="Fetch")

        self.counter = 0
        self.output_dir = "output/real_world"

        # initialize network
        self.text_prompt = "table .  door . chair ."
        self.gdino = GroundingDINOObjectPredictor()
        self.SAM = SegmentAnythingPredictor()

        self.image_pub = rospy.Publisher("seg_image", Image, queue_size=10)
        
        self.graph = Graph()
        self.pose_list = {"table":[], "chair":[], "door":[]}
        self.threshold = {"table": 2, "chair":0.6, "door": 2}
        self.marker_pub = rospy.Publisher("graph_nodes", MarkerArray, queue_size=10)
        time.sleep(5)
        self.marker_pub = rospy.Publisher("graph_nodes", MarkerArray, queue_size=10)

    def create_marker(self, pose, category, node_id):
        """
        Creates a Marker for a graph node to be displayed in RViz.
        """
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = category
        marker.id = node_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = pose[0]
        marker.pose.position.y = pose[1]
        marker.pose.position.z = pose[2] * 0  
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.3 
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        marker.color.a = 1.0  

        if category == "table":
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
        elif category == "chair":
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
        elif category == "door":
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0

        return marker
    
    def publish_graph_to_rviz(self):
        """
        Publishes the graph nodes as markers in RViz.
        """
        marker_array = MarkerArray()
        node_id = 0

        for node, data in self.graph.nodes(data=True):
            pose = data["pose"]
            category = data["category"]

            marker = self.create_marker(pose, category, node_id)
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

            # OpenCV 이미지를 PIL로 변환
            img_pil = PILImg.fromarray(cv2.cvtColor(im_color, cv2.COLOR_BGR2RGB))

            bboxes, phrases, gdino_conf = self.gdino.predict(img_pil, self.text_prompt, 0.55, 0.55)
            bboxes, gdino_conf, phrases, flag = filter(bboxes, gdino_conf, phrases, 1, 0.8, 0.8, 0.8, 0.01, True)
            if flag:
                continue

            if len(phrases) == 0:
                print(f"Skipping zero phrases\n")
                continue 

            w, h = img_pil.size
            image_pil_bboxes = self.gdino.bbox_to_scaled_xyxy(bboxes, w, h)

            image_pil_bboxes, masks = self.SAM.predict(img_pil, image_pil_bboxes)

            print(masks.shape)
            image_pil_bboxes, index = filter_large_boxes(image_pil_bboxes, w, h, threshold=0.5)
            masks = masks[index]

            mask_array = masks.cpu().numpy()
            phrase_iter_ = {"table": 0, "door": 0, "chair": 0}

            for i, mask in enumerate(mask_array):
                mask = mask[0]
                pose = pose_in_map_frame(RT_camera, RT_base, depth_img, segment=mask)
                print(f"pose {pose} class {phrases[i]}")
                if pose is None:
                    continue
                self.pose_list[phrases[i]], _is_nearby = is_nearby_in_map(
                    self.pose_list[phrases[i]], pose, threshold=self.threshold[phrases[i]]
                )
                if not _is_nearby:
                    print(f"Adding node")
                    self.graph.add_node(
                        f"{phrases[i]}_{iter_}_{phrase_iter_[phrases[i]]}",
                        id=f"{phrases[i]}_{iter_}_{phrase_iter_[phrases[i]]}",
                        pose=pose,
                        robot_pose=RT_base.tolist(),
                        category=phrases[i],
                    )
                    phrase_iter_[phrases[i]] += 1
                self.pose_list[phrases[i]].append(pose)
                
            mask = combine_masks(masks[:, 0, :, :]).cpu().numpy()
            gdino_conf = gdino_conf[index]
            ind = np.where(index)[0]
            phrases = [phrases[i] for i in ind]

            # Bounding box 및 마스크 추가
            bbox_annotated_pil = annotate(
                overlay_masks(img_pil, masks), image_pil_bboxes, gdino_conf, phrases
            )

            im_label = np.array(bbox_annotated_pil)

            # OpenCV 형식으로 변환 후 ROS 메시지로 변환
            rgb_msg = self.bridge.cv2_to_imgmsg(cv2.cvtColor(im_label, cv2.COLOR_RGB2BGR), encoding="bgr8")
            rgb_msg.header.stamp = rgb_frame_stamp
            rgb_msg.header.frame_id = rgb_frame_id
            self.image_pub.publish(rgb_msg)

            self.publish_graph_to_rviz()
            iter_ += 1


if __name__ == "__main__":
    robokit_instance = robokitRealtime()
    robokit_instance.run_network()
    print(f"Closing script! Saving graph")
    save_graph_json(robokit_instance.graph, file="graph.json")

