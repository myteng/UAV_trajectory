import ast

import numpy as np
import pandas as pd
from shapely.geometry import Polygon, LineString


# 创建建筑的多边形面
def create_building_faces(base_vertices, height):
    """
    创建建筑的多边形面

    参数:
    base_vertices (list of tuples): 建筑底面的顶点坐标，格式为[(x1, y1), (x2, y2), ...]
    height (float): 建筑的高度

    返回:
    dict: 包含建筑底面、顶面和侧面的多边形面
    """
    # 将底面顶点转化为numpy数组
    base_vertices = np.array(base_vertices)

    # 创建顶面顶点
    top_vertices = base_vertices.copy()
    top_vertices = np.hstack((top_vertices, np.full((top_vertices.shape[0], 1), height)))

    # 底面顶点添加高度 0
    base_vertices = np.hstack((base_vertices, np.zeros((base_vertices.shape[0], 1))))

    # 创建建筑的底面和顶面
    base_polygon = Polygon(base_vertices)
    top_polygon = Polygon(top_vertices)

    # 创建侧面
    side_polygons = []
    num_vertices = len(base_vertices)
    for i in range(num_vertices):
        j = (i + 1) % num_vertices
        side_polygon = Polygon([base_vertices[i], base_vertices[j], top_vertices[j], top_vertices[i]])
        side_polygons.append(side_polygon)

    # 返回底面、顶面和侧面
    return {
        'base': base_polygon,
        'top': top_polygon,
        'sides': side_polygons
    }


# 判断线段是否经过多边形区域
def line_segment_intersects_polygon(line_start, line_end, polygon):
    line_2d = LineString([line_start, line_end])
    polygon_2d = Polygon([(p[0], p[1]) for p in polygon.exterior.coords])
    return line_2d.intersects(polygon_2d)


# 判断线段是否与建筑相交
def line_segment_intersects_building(path, line_start, line_end):
    """
    判断线段是否与建筑相交

    参数:
    base_vertices (list of tuples): 建筑底面的顶点坐标，格式为[(x1, y1), (x2, y2), ...]
    height (float): 建筑的高度
    line_start (tuple): 线段起点的三维坐标 (x, y, z)
    line_end (tuple): 线段终点的三维坐标 (x, y, z)

    返回:
    bool: 线段是否与建筑相交
    """
    # 初始化统计变量
    intersects_building_num = 0
    building_info = []

    # 加载 CSV 文件
    file_path = '/Users/tengman/Documents/Python/PycharmProjects/uav_sequence_offloading/building_dataset/' + path  # 替换为实际文件路径
    df = pd.read_csv(file_path)
    # 获取建筑顶点坐标和高度
    for index, row in df.iterrows():
        ID = row['ID']
        height = row['height']
        base_vertices = ast.literal_eval(row['x_y'])
        building_faces = create_building_faces(base_vertices, height)
        # print(building_faces)

        # 初始化建筑是否与线段相交的标志
        is_intersects = False

        # 检查底面和顶面
        if line_segment_intersects_polygon(line_start, line_end, building_faces['base']) or line_segment_intersects_polygon(
                line_start, line_end, building_faces['top']):
            is_intersects = True
        else:
            # 检查侧面
            for side in building_faces['sides']:
                if line_segment_intersects_polygon(line_start, line_end, side):
                    is_intersects = True
                    break  # 一旦发现相交即可退出循环
        # 如果建筑与线段相交，更新统计信息
        if is_intersects:
            intersects_building_num += 1
            building_info.append({"ID": ID, "Height": height})
    return intersects_building_num, building_info


# # 示例调用
# point_a = np.array([0,0,0])
# point_b = np.array([100,100,100])
# print(point_a,point_b)
# print(type(point_b))
#
#
# # point_a = ({"location":[0,0,0], "tr"})
# intersects_building_num, building_info = line_segment_intersects_building(point_a, point_b)
#
# print(f"Line segment intersects building, number is {intersects_building_num}, informance is {building_info}:",)

