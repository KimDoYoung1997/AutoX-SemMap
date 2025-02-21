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
from utils2 import (
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
        # self.marker_pub = rospy.Publisher("graph_nodes", MarkerArray, queue_size=10)
        # time.sleep(5)
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
            # ImageListener의 callback 스레드가 데이터를 Write  동안 해당 스레드에서 데이터를 읽는걸 방지 (Data Race 방지)
            with lock:
                if self.listener.im is None:    # RGB image 데이터가 없으면 넘어감
                    continue
                im_color = self.listener.im.copy()
                depth_img = self.listener.depth.copy()
                rgb_frame_id = self.listener.rgb_frame_id
                rgb_frame_stamp = self.listener.rgb_frame_stamp
                RT_camera, RT_base = self.listener.RT_camera, self.listener.RT_base
            
            print("===========================================")

            # OpenCV 이미지를 PIL로 변환
                # GroundingDINO와 SAM 모델이 PIL(Python Imaging Library) 이미지 형식을 입력으로 기대하기 때문
            img_pil = PILImg.fromarray(cv2.cvtColor(im_color, cv2.COLOR_BGR2RGB)) # bgr -> rgb 변환

            # - bboxes: 검출된 객체들의 바운딩 박스 좌표
            # - phrases: 검출된 객체들의 텍스트 레이블
            # - gdino_conf: 바운딩 박스 신뢰도  
            bboxes, phrases, gdino_conf = self.gdino.predict(
                img_pil,           # PIL 형식의 입력 이미지
                self.text_prompt,  # "table . door . chair ." - 검출하고자 하는 객체들
                0.55,             # box_threshold: 바운딩 박스 신뢰도 임계값
                0.55              # text_threshold: 텍스트 매칭 신뢰도 임계값
            )
            # filter 함수는 객체 검출 결과에서 false positive를 제거하기 위한 여러 필터링 조건을 적용하는 함수입니다.
            # 예를 들어, 바운딩 박스의 너비와 높이가 이미지의 일정 비율 이상/이하 이면 제거합니다.
            # 또한, 바운딩 박스의 신뢰도가 낮으면 제거합니다.   
            # 문의 경우 높이/너비 비율이 1.7 미만이면 제거
            # 문의 면적이 전체 이미지의 4% 미만이면 제거
            # def filter(bboxes, conf_list, phrases ,conf_bound, yVal, precentWidth=0.5, precentHeight=0.5, precentArea=0.05, filterChoice=True)
            bboxes, gdino_conf, phrases, flag = filter(bboxes, gdino_conf, phrases, 1, 0.8, 0.8, 0.8, 0.01, True)
            
            # flag (boolean): indicating if there are any detections left after filter
                # false = no detections left (breaks loop)
                # true =  some detections left (continuted) 
            rospy.loginfo(f"flag: {flag}, Keep going!")
            if flag:
                continue
            # phrases (list): Filtered list of phrases
            rospy.loginfo(f"phrases: {phrases}, number of Filtered list of phrases: {len(phrases)}")
            if len(phrases) == 0:
                print(f"Skipping zero phrases\n")
                continue 

            w, h = img_pil.size
            
            # 이 함수는 바운딩 박스 좌표를 이미지의 실제 크기에 맞게 스케일링하는 작업을 수행합니다.
            # 일반적으로 객체 검출 모델(여기서는 GroundingDINO)은 바운딩 박스 좌표를 정규화된 형태(0~1 사이의 값)로 출력합니다. 
            # 이를 실제 이미지 픽셀 좌표로 변환해야 합니다.            image_pil_bboxes = self.gdino.bbox_to_scaled_xyxy(bboxes, w, h)
            image_pil_bboxes = self.gdino.bbox_to_scaled_xyxy(bboxes, w, h)
            
            # rospy.loginfo(f"GroundingDino Output : image_pil_bboxes: {image_pil_bboxes}")
            
            # SAM(Segment Anything Model)의 masks는 각 객체의 세그멘테이션 마스크를 나타냅니다
            image_pil_bboxes, masks = self.SAM.predict(img_pil, image_pil_bboxes)   
            # rospy.loginfo(f"SAM Output : image_pil_bboxes: {image_pil_bboxes}, masks: {masks}")
            
            # masks.shape  # [N, 1, H, W] N: 검출된 객체의 수, 1: 채널 수, H 480: 이미지 높이, W 640: 이미지 너비
            # 예시 
            # 3개의 객체가 검출되었다고 가정
            # masks = torch.tensor([
            #     [  # 첫 번째 객체 (예: 테이블)
            #         [[0, 0, 0, 0, 0],
            #         [0, 1, 1, 1, 0],
            #         [0, 1, 1, 1, 0],
            #         [0, 0, 0, 0, 0]]
            #     ],
            #     [  # 두 번째 객체 (예: 의자)
            #         [[0, 0, 0, 0, 0],
            #         [0, 0, 1, 0, 0],
            #         [0, 1, 1, 1, 0],
            #         [0, 0, 0, 0, 0]]
            #     ],
            #     [  # 세 번째 객체 (예: 문)
            #         [[1, 1, 0, 0, 0],
            #         [1, 1, 0, 0, 0],
            #         [1, 1, 0, 0, 0],
            #         [1, 1, 0, 0, 0]]
            #     ]
            # ])
            rospy.loginfo(f"masks.shape: {masks.shape}")

            # filter_large_boxes 함수 : Filter out large boxes from a list of bounding boxes based on a threshold.
            # filter_large_boxes 함수는 각 바운딩 박스의 면적을 계산하고, 이미지 전체 면적의 50%(threshold)보다 큰 박스를 제거하고, 필터링된 바운딩 박스 리스트와 해당 인덱스를 반환
            image_pil_bboxes, index = filter_large_boxes(image_pil_bboxes, w, h, threshold=0.5)
            
            # filter_large_boxes 함수에서 너무 큰 바운딩 박스를 필터링한 결과에 맞춰서, 해당하는 마스크들도 같이 필터링하는 역할
            masks = masks[index]

            mask_array = masks.cpu().numpy()    # masks PyTorch 텐서를 NumPy 배열로 변환
            phrase_iter_ = {"table": 0, "door": 0, "chair": 0}  # 왜 있는지 모르겠음.
            # 아래 for 반복문은 검출된 각 객체의 3D 위치를 계산하고, 이를 그래프에 노드로 추가하는 부분입니다
            for i, mask in enumerate(mask_array):
                rospy.loginfo(f"22222222222222222222222222222222222222222222222222222222222222")
  
                # true_indices = np.argwhere(mask[0])  # [y, x] 좌표 반환
                # rospy.loginfo(f"Mask {i} True pixels at: {true_indices}")
                # rospy.loginfo(f"Number of True pixels in mask[{i}]: {len(true_indices)}")
                # rospy.loginfo(f"Number of All pixels in mask[{i}]: {480 * 640}")

                
                mask = mask[0]
                rospy.loginfo(f"mask[0].shape: {mask.shape}")
                
                pose = pose_in_map_frame(RT_camera, RT_base, depth_img, segment=mask)
                rospy.logwarn(f"pose {pose} class_id : {i} : {phrases[i]}")    # pose None class chair
                if pose is None:
                    continue
                
                # # phrases[i]는 현재 검출된 객체의 종류 (예: "door")
                self.pose_list[phrases[i]], _is_nearby = is_nearby_in_map(
                    self.pose_list[phrases[i]],                     # 현재 기존 클래스의 보유 pose 리스트
                    pose,                                           # 현재 새로 검출된 pose       
                    threshold=self.threshold[phrases[i]]            # 현재 클래스의 threshold , 클래스별 임계값
                )
                rospy.loginfo(f"1. phrases[i]: {phrases[i]}")
                # rospy.logerr(f"1. self.pose_list[phrases[i]]: {self.pose_list[phrases[i]]}")
                rospy.logwarn(f"1. len(self.pose_list[phrases[i]]): {len(self.pose_list[phrases[i]])}")
                
                # 새로운 객체일 때만 그래프에 노드 추가
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
                    # 3. 새로운 객체일 때만 pose_list에 추가
                self.pose_list[phrases[i]].append(pose)
                rospy.loginfo(f"2. phrases[i]: {phrases[i]}")
                # rospy.logerr(f"2.self.pose_list[phrases[i]]: {self.pose_list[phrases[i]]}")
                rospy.logwarn(f"2. len(self.pose_list[phrases[i]]): {len(self.pose_list[phrases[i]])}")
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
    try:
        robokit_instance.run_network()  # 네트워크 실행
    except KeyboardInterrupt:
        print("\n Ctrl+C 감지됨. 그래프 저장 중...")
    finally:
        print(f" 그래프 저장 완료: graph.json")
        save_graph_json(robokit_instance.graph, file="graph.json")

