from typing import List

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, LineString
import math
import random

from envs.map import Map
from envs.obstacle import ObstacleMap
from parameter.paramEnv import args_env


class Node:
    def __init__(self, pos, parent=None):
        self.pos = np.array(pos)  # 当前节点位置
        self.parent = parent  # 父节点，用于回溯路径
        self.cost = 0.0 if parent is None else parent.cost + np.linalg.norm(self.pos - parent.pos)  # 当前节点从起点累计的代价


class InformedRRTStar:
    def __init__(self, start, goal, obstacle_list, goal_sample_rate=0.05, max_iter=1000):
        self.start = Node(start)
        self.goal = Node(goal)
        self.obstacle_list = obstacle_list  # 使用 shapely Polygon 表示的障碍物列表

        # 地图边界
        self.map = Map()
        self.minx = 0
        self.miny = 0
        self.maxx = self.map.ranges_x
        self.maxy = self.map.ranges_y
        # self.minx, self.miny, self.maxx, self.maxy = map_bounds  # 地图边界

        self.step_len = args_env.vels_uav  # 每次树扩展的步长
        self.goal_sample_rate = goal_sample_rate  # 使用启发式（goal采样）的概率
        self.search_radius = 3.0
        self.max_iter = max_iter  # 最大迭代次数
        self.node_list = [self.start]  # 节点树初始化
        self.c_best = float('inf')  # 当前最短路径长度
        self.path = None  # 最优路径

    def planning(self):
        for _ in range(self.max_iter):
            # 启发式采样（若已有可行路径则启用椭球采样）
            if random.random() < self.goal_sample_rate and self.path:
                # 在椭球体中采样
                rnd = self.informed_sample()
            else:
                # 随机采样整个空间
                rnd = np.array([random.uniform(self.minx, self.maxx), random.uniform(self.miny, self.maxy)])

            # 找到最近的节点
            nearest_node = min(self.node_list, key=lambda node: np.linalg.norm(node.pos - rnd))

            # 沿着方向扩展一定步长
            direction = rnd - nearest_node.pos
            length = np.linalg.norm(direction)
            if length == 0:
                continue
            direction = direction / length
            new_pos = nearest_node.pos + self.step_len * direction

            # 若路径与障碍物冲突，则跳过
            if self.is_collision(nearest_node.pos, new_pos):
                continue

            # 生成新节点
            new_node = Node(new_pos, nearest_node)

            # 寻找一定半径范围内的邻居节点
            near_nodes = [node for node in self.node_list if np.linalg.norm(node.pos - new_node.pos) <= self.search_radius]
            min_cost = new_node.cost
            best_parent = nearest_node

            # 从邻居中选择代价最小的父节点
            for node in near_nodes:
                temp_cost = node.cost + np.linalg.norm(node.pos - new_node.pos)
                if temp_cost < min_cost and not self.is_collision(node.pos, new_node.pos):
                    best_parent = node
                    min_cost = temp_cost

            # 设置最优父节点和代价
            new_node.parent = best_parent
            new_node.cost = min_cost
            self.node_list.append(new_node)

            # 重接操作：检查是否可以通过新节点让邻居节点代价更小
            for node in near_nodes:
                temp_cost = new_node.cost + np.linalg.norm(node.pos - new_node.pos)
                if temp_cost < node.cost and not self.is_collision(new_node.pos, node.pos):
                    node.parent = new_node
                    node.cost = temp_cost

            # 如果新节点已经足够接近目标点并无碰撞，更新最短路径
            if np.linalg.norm(new_node.pos - self.goal.pos) <= self.step_len and not self.is_collision(new_node.pos, self.goal.pos):
                self.goal.parent = new_node
                self.goal.cost = new_node.cost + np.linalg.norm(new_node.pos - self.goal.pos)
                self.path = self.extract_path(self.goal)
                self.c_best = self.goal.cost

        return self.path

    def is_collision(self, p1, p2):
        """判断线段是否与任何障碍物相交"""
        line = LineString([p1, p2])
        for poly in self.obstacle_list:
            if line.intersects(poly):
                return True
        return False

    def informed_sample(self):
        """在椭球体中采样，用于收缩采样空间"""
        c_min = np.linalg.norm(self.start.pos - self.goal.pos)
        if self.c_best == float('inf'):
            return np.array([random.uniform(self.minx, self.maxx), random.uniform(self.miny, self.maxy)])

        center = (self.start.pos + self.goal.pos) / 2  # 椭球中心
        a1 = (self.goal.pos - self.start.pos) / c_min  # 单位方向向量

        # 构建旋转矩阵 C，使采样椭球与目标方向对齐
        M = np.outer(a1, np.array([1.0, 0.0]))
        U, _, Vt = np.linalg.svd(M)
        C = np.dot(U, Vt)

        # 长短轴长度
        r1 = self.c_best / 2.0
        r2 = math.sqrt(self.c_best ** 2 - c_min ** 2) / 2.0

        # 在单位圆内采样，然后映射到椭球
        while True:
            x = np.random.normal(0, 1, 2)
            if np.linalg.norm(x) <= 1:
                break

        rnd = np.dot(C, np.diag([r1, r2]).dot(x)) + center
        return rnd

    def extract_path(self, node):
        """从目标节点反向提取完整路径"""
        path = []
        while node:
            path.append(node.pos)
            node = node.parent
        return path[::-1]  # 从起点到终点


# 示例用法（若作为独立脚本运行）
if __name__ == '__main__':
    # 设置起点、终点、障碍物和地图边界
    start = (50, 50)
    goal = (150, 150)
    # obstacles = [
    #     Polygon([(5, 5), (5, 15), (6, 15), (6, 5)]),         # 垂直矩形障碍物
    #     Polygon([(12, 0), (12, 8), (13, 8), (13, 0)])        # 左右矩形障碍物
    # ]
    obs_map = ObstacleMap()
    obstacles: List[Polygon] = [obs.geometry for obs in obs_map.obstacles]
    bounds = (0, 0, 200, 200)  # 地图大小

    # 执行路径规划
    planner = InformedRRTStar(start, goal, obstacles)
    path = planner.planning()

    # 可视化路径和障碍物
    fig, ax = plt.subplots()
    for poly in obstacles:
        x, y = poly.exterior.xy
        ax.fill(x, y, color='gray')  # 灰色为障碍物
    ax.plot(start[0], start[1], "go")  # 起点
    ax.plot(goal[0], goal[1], "ro")   # 终点
    if path:
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], 'b--')  # 蓝色虚线为路径
    plt.axis("equal")
    plt.title("Informed RRT* Path Planning with Comments")
    plt.show()
