import numpy as np
import os
import cv2
import yaml
from numpy.linalg import norm
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import networkx as nx
from networkx.readwrite import json_graph
import json

intrinsics = [
    [574.0527954101562, 0.0, 319.5],
    [0.0, 574.0527954101562, 239.5],
    [0.0, 0.0, 1.0],
]
# intrinsics = [[554.254691191187, 0.0, 320.5],[0.0, 554.254691191187, 240.5], [0.0, 0.0, 1.0]]
fx = intrinsics[0][0]
fy = intrinsics[1][1]
px = intrinsics[0][2]
py = intrinsics[1][2]


def compute_xyz(depth_img, fx, fy, px, py, height, width):
    """
    깊이 이미지의 각 픽셀을 3D 좌표로 변환

    Args:
        depth_img (ndarray): [H, W] 형태의 깊이 이미지
        fx (float): 카메라 초점 거리 x
        fy (float): 카메라 초점 거리 y
        px (float): 카메라 주점 x 좌표
        py (float): 카메라 주점 y 좌표
        height (int): 이미지 높이
        width (int): 이미지 너비

    Returns:
        ndarray: [H, W, 3] 형태의 3D 좌표 배열, 각 픽셀마다 (x, y, z) 좌표
    """
    indices = np.indices((height, width), dtype=np.float32).transpose(1, 2, 0)  # indices된 [2,H,W] 를 -> [H, W, 2] 형태로 transpose
    print(f"indices.shape: {indices.shape}")    # [H, W, 2] 형태
    z_e = depth_img
    print(f"z_e.shape: {z_e.shape}")    # [H, W] 형태
    x_e = (indices[..., 1] - px) * z_e / fx # # x 좌표 계산: (u - cx) * z / fx , 각 픽셀의 열 인덱스(x) 정보
    y_e = (indices[..., 0] - py) * z_e / fy # # y 좌표 계산: (v - cy) * z / fy , 각 픽셀의 행 인덱스(y) 정보
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1)   # 
    print(f"xyz_img.shape: {xyz_img.shape}")
    return xyz_img


def pose_to_map_pixel(map_metadata, pose):
    pose_x = pose[0]
    pose_y = pose[1]

    map_pixel_x = int((pose_x - map_metadata["origin"][0]) / map_metadata["resolution"])
    map_pixel_y = int((pose_y - map_metadata["origin"][1]) / map_metadata["resolution"])

    return [map_pixel_x, map_pixel_y]


def pose_along_line(pose1, pose2, distance=2):
    '''
    creates a new pose that is at the specified distance from pose1
    along the line from pose1 to pose2
    '''
    pose2 = pose2[0:3,3]
    difference_vector = pose2 - pose1
    unit_vector = difference_vector / norm(difference_vector)
    new_pose = pose1 + unit_vector * distance

    return new_pose


def read_map_image(map_file_path):
    assert os.path.exists(map_file_path)
    if map_file_path.endswith(".pgm"):
        map_image = cv2.imread(map_file_path)
    else:
        map_image = cv2.imread(map_file_path)

    return map_image


def read_map_metadata(metadata_file_path):
    assert os.path.exists(metadata_file_path)
    assert metadata_file_path.endswith(".yaml")
    with open(metadata_file_path, "r") as file:
        metadata = yaml.safe_load(file)
    file.close()
    return metadata


def display_map_image(map_image, write=False):
    width, height, _ = map_image.shape
    cv2.namedWindow("Map Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Map Image", width, height)
    if write:
        cv2.imwrite("map_image.png", map_image)
    cv2.imshow("Map Image", map_image)
    cv2.waitKey(0)


def is_nearby(pose1, pose2, threshold=0.5):
    if norm((pose1[0] - pose2[0], pose1[1] - pose2[1])) < threshold:
        return True


def normalize_depth_image(depth_array, max_depth):
    depth_image = (max_depth - depth_array) / max_depth
    depth_image = depth_image * 255
    return depth_image.astype(np.uint8)


def denormalize_depth_image(depth_image, max_depth):

    depth_array = max_depth * (1 - (depth_image / 255))
    # print(f"max {depth_array.max()}")
    return depth_array.astype(np.float32)

def get_fov_points_in_baselink(depth_array, RT_camera):
        mask1 = np.isnan(depth_array)
        depth_array[mask1] = 0.0
        xyz_array = compute_xyz(
            depth_array, fx, fy, px, py, depth_array.shape[0], depth_array.shape[1]
        )
        xyz_array = xyz_array.reshape((-1, 3))

        mask = ~(np.all(xyz_array == [0.0, 0.0, 0.0], axis=1))
        xyz_array = xyz_array[mask]

        xyz_base = np.dot(RT_camera[:3, :3], xyz_array.T).T
        xyz_base += RT_camera[:3, 3]

        min_x = np.min(xyz_base[:,0])
        max_x = np.max(xyz_base[:,0])
        min_y = np.min(xyz_base[:,1])
        max_y = np.max(xyz_base[:,1])

        return [[0,0,0],[max_x,min_y,0], [max_x, max_y,0]]

def get_fov_points_in_map(depth_array, RT_camera, RT_base):
        mask1 = np.isnan(depth_array)
        depth_array[mask1] = 0.0
        xyz_array = compute_xyz(
            depth_array, fx, fy, px, py, depth_array.shape[0], depth_array.shape[1]
        )
        xyz_array = xyz_array.reshape((-1, 3))

        mask = ~(np.all(xyz_array == [0.0, 0.0, 0.0], axis=1))
        xyz_array = xyz_array[mask]

        xyz_base = np.dot(RT_camera[:3, :3], xyz_array.T).T
        xyz_base += RT_camera[:3, 3]

        min_x = np.min(xyz_base[:,0])
        max_x = np.max(xyz_base[:,0])
        min_y = np.min(xyz_base[:,1])
        max_y = np.max(xyz_base[:,1])

        points_baselink = [[0,0,0],[max_x,min_y,0], [max_x, max_y,0]]
        points_map = np.dot(RT_base[:3,:3], np.array(points_baselink).T).T + RT_base[:3,3]

        return points_map.tolist()

def pose_in_map_frame(RT_camera, RT_base, depth_array, segment=None):
    """
    깊이 이미지와 세그멘테이션 마스크를 사용하여 객체의 3D 위치를 맵 좌표계로 변환합니다.

    Args:
        RT_camera (np.ndarray): Camera (rgb_optical_frame)에서 base_link로의 변환 행렬 [4, 4]
        RT_base (np.ndarray): base_link에서 map 좌표계로의 변환 행렬 [4, 4]
        depth_array (np.ndarray): 깊이 이미지 [H, W]
        segment (np.ndarray, optional): SAM에서 생성된 세그멘테이션 마스크 [H, W]. 
                                      True 값은 객체 영역을 나타냄. Defaults to None.

    Returns:
        Union[List[float], None]: 객체의 맵 좌표계 상의 평균 위치 [x, y, z].
                                 유효한 깊이값이 없는 경우 None 반환.

    Process:
        1. 세그멘테이션 마스크로 깊이 이미지 필터링 (segment가 제공된 경우)
        2. NaN 값을 0으로 처리
        3. 깊이 이미지를 3D 포인트 클라우드로 변환
        4. 0이 아닌 포인트들만 선택
        5. 카메라 -> 베이스 -> 맵 좌표계로 순차적 변환
        6. 포인트들의 평균 위치 계산

    Note:
        - depth_array의 값은 미터 단위여야 함
        - RT_camera와 RT_base는 4x4 동차 변환 행렬이어야 함
        - 반환된 위치는 맵 좌표계 기준
    """
    if segment is not None:
        # depth_array: 깊이 이미지
        print(f"depth_array.max(): {depth_array.max()}")    # 약 5m 정도
        
        # 마스크를 통한 깊이값 필터링, Segmentation 마스크에 해당하는(True) 부분만 깊이값을 유지
        # 이 시점에서:
            # - 마스크가 True(1)이더라도 깊이값이 0이면 결과는 0
            # - 마스크가 True이고 깊이값이 유효하면 깊이값 유지
            # - 마스크가 False(0)이면 무조건 0
        depth_array = (depth_array) * (segment / 1)
        # depth_array = (depth_array+0.001) * (segment / 1)  <-  rue(1)이더라도 깊이값이 0이면 결과는 0 인걸 방지함, 따라서 이제 마스크의 True 픽셀과 유효한 깊이값을 가진 픽셀의 수가 일치합니다
        print(f"segment 이미지에 의해 필터링된 depth_array 픽셀 수: {len(np.argwhere(depth_array))}")
        
        # NaN 개수 확인
        nan_count = np.sum(np.isnan(depth_array))
        total_count = depth_array.size
        print(f"NaN count: {nan_count} out of {total_count} pixels ({(nan_count/total_count)*100:.2f}%)")

    #TODO: if depth is not normalized, then we need to remoev nans in the read image 
    # depth_array[np.isnan(depth_array)] = 0.0
    # 깊이 이미지에서 NaN 값을 0으로 처리
    mask1 = np.isnan(depth_array)
    depth_array[mask1] = 0.0
    

    if depth_array.max() == 0.0:
        print("depth_array.max() == 0.0 , Return None")
        return None
    else:
        # compute_xyz 함수는 깊이 이미지의 각 픽셀을 3D 좌표로 변환하여, ndarray: [H, W, 3] 형태의 3D 좌표 배열, 각 픽셀마다 (x, y, z) 좌표
        xyz_array = compute_xyz(
            depth_array,                        # 필터링된 깊이 이미지
            fx, fy,                             # 카메라 초점 거리
            px, py,                             # 카메라 주점
            depth_array.shape[0],               # 이미지 height
            depth_array.shape[1]                # 이미지 width
        )
        
        xyz_array = xyz_array.reshape((-1, 3))
        print(f"xyz_array.shape: {xyz_array.shape}") # 480 * 640 [H*W, 3] 형태로 변환
        mask = ~(np.all(xyz_array == [0.0, 0.0, 0.0], axis=1))  # [0,0,0] 포인트를 제외한 나머지 포인트만 선택
        xyz_array = xyz_array[mask]                             # 유효한 포인트만 남김
        print(f"xyz_array[mask].shape: {xyz_array.shape}") 

        xyz_base = np.dot(RT_camera[:3, :3], xyz_array.T).T     # 3X3 행렬과 3XN 행렬의 곱셈, 결과는 3XN 행렬
        xyz_base += RT_camera[:3, 3]

        print(f"mean pose base link {np.mean(xyz_base, axis=0)}") # 베이스 좌표계에서의 평균 위치
        print(f"xyz_base.shape: {xyz_base.shape}")
        mean_pose_base = np.mean(xyz_base, axis=0)
        print(f"mean pose wrt base link {mean_pose_base}")
        xyz_map = np.dot(RT_base[:3, :3], xyz_base.T).T
        xyz_map += RT_base[:3, 3]
        mean_pose = np.mean(xyz_map, axis=0)
        print(f"mean pose wrt map {mean_pose}")
        # mean_pose = pose_along_line( mean_pose, RT_base)
        return mean_pose.tolist()


def is_nearby_in_map(pose_list, node_pose, threshold=0.5):
    """
    새로 검출된 객체가 기존 객체들과 충분히 가까운지 확인합니다.
    2D 평면(x-y)상에서의 거리만 고려합니다.
    is_nearby_in_map 함수는 중복 검출을 방지하기 위해 사용됩니다.
    이 함수가 없다면:
        같은 물체를 계속 새로운 물체로 등록
        맵이 중복된 객체들로 채워짐

    Args:
        pose_list (List[List[float]]): 기존에 검출된 객체들의 3D 위치 목록 [[x1,y1,z1], [x2,y2,z2], ...]
        node_pose (List[float]): 새로 검출된 객체의 3D 위치 [x,y,z]
        threshold (float, optional): 같은 객체로 판단할 최대 거리(미터). Defaults to 0.5.

    Returns:
        Tuple[List[List[float]], bool]: 
            - 업데이트된 pose_list (새 객체가 추가되었을 수 있음)
            - 근처에 객체가 있는지 여부 (True: 있음, False: 없음)

    Example:
        >>> pose_list = [[1.0, 1.0, 0.0], [2.0, 2.0, 0.0]]
        >>> new_pose = [1.1, 1.1, 0.0]
        >>> pose_list, is_nearby = is_nearby_in_map(pose_list, new_pose, 0.5)
        >>> print(is_nearby)  # True (1m 이내에 기존 객체 있음)
    """

    # 기존 객체가 없으면 바로 False 반환
    if len(pose_list) == 0:
        print(f"pose_list 가 비어있어서 새로운 객체 추가 안함")
        return pose_list, False

    # if len(pose_list) == 0:
    #     pose_list.append(node_pose)  # 여기를 수정!
    #     print(f"pose_list가 비어있어서 첫 객체로 추가함")
    #     return pose_list, False


    # NumPy 배열로 변환
    pose_array = np.array(pose_list)        # 기존 객체들의 위치
    node_pose_array = np.array([node_pose]) # 새 객체의 위치

    # x-y 평면상의 유클리드 거리 계산
    distances = np.linalg.norm((pose_array[:, 0:2] - node_pose_array[:, 0:2]), axis=1)

    # threshold 거리 이내에 객체가 있는지 확인
    if np.any(distances < threshold):
        print(f"distances 내에 객체가 있어서 새로 추가하지 않음 ,distances < threshold: {bool(np.any(distances < threshold))}")
        return pose_list, True  # 근처에 객체가 있음, # 기존 pose_list 유지, nearby=True
    else:
        print(f"distances 내에 객체가 없어서 새로 추가함, distances < threshold: {bool(np.any(distances < threshold))}")
        pose_list.append(node_pose)  # 새 객체 추가
        return pose_list, False      # 근처에 객체가 없음 # 업데이트된 pose_list 반환, nearby=False

def save_graph_json(graph, file="graph.json"):
    '''
    input graph \n
    save graph to graph.json
    '''
    file = file
    data_to_save = json_graph.node_link_data(graph)
    with open(file, "w") as file:
        json.dump(data_to_save, file, indent=4)
        file.close()
    print(f"-=---------------------")


def read_graph_json(file="graph.json"):
    with open(file, "r") as file:
        data = json.load(file)
        file.close()
    # print(data)
    graph = json_graph.node_link_graph(data)
    return graph


def read_and_visualize_graph(map_file_path, map_metadata_filepath, on_map=False, catgeories=[], graph=None):
    if graph is None:
        graph = read_graph_json()
    else:
        graph = graph
    color_palette = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
    if not on_map:
        pos = nx.spring_layout(graph)
        nx.draw(graph, pos, with_labels=True)
        plt.show()
    else:
        # c ncj
        map_image = read_map_image(map_file_path)
        map_metadata = read_map_metadata(map_metadata_filepath)
        for node, data in graph.nodes(data=True):
            if data["category"] in catgeories:
                x, y = pose_to_map_pixel(map_metadata, data["pose"])
                map_image[
                    y - 10 // 2 : y + 10 // 2,
                    x - 10 // 2 : x + 10 // 2,
                    :,
                ] = color_palette[catgeories.index(data["category"])]
        display_map_image(map_image, write=True)

def plot_point_on_map(map_file_path, map_metadata_filepath, position):
    map_image = read_map_image(map_file_path)
    map_metadata = read_map_metadata(map_metadata_filepath)
    x, y = pose_to_map_pixel(map_metadata, position)
    map_image[
        y - 10 // 2 : y + 10 // 2,
        x - 10 // 2 : x + 10 // 2,
        :,
    ] = [0,0,255]
    display_map_image(map_image, write=False)

def visualize_graph(graph):
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True)
    plt.show()


if __name__ == "__main__":
    # a=compute_xyz(np.array([[0,0,0],[0,0,0],[0,0,0]]), fx,fy,px,py, 3,3)
    # a=a.reshape((-1,3))
    # print(a)
    # save_graph_json()
    graph = read_graph_json()
    read_and_visualize_graph("map.png","map.yaml", on_map=True, catgeories=["table", "chair", "door"], graph=graph)
