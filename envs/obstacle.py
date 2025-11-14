import random
from typing import Tuple, List
from shapely.geometry import Point, box, LineString

from envs.map import Map
from parameter.paramEnv import args_env


# 用于将3D点转换到2D平面
def union_xy(point3d: Tuple[float, float, float]) -> Tuple[float, float]:
    return point3d[0], point3d[1]


class Obstacle:
    def __init__(self, shape: str, center: Tuple[float, float, float], size: Tuple[float, float]):
        """
        shape: 'circle' 或 'rectangle'
        center: (x, y, height)
        size:
            - circle: (radius, 0)
            - rectangle: (width, height)
        """
        self.shape = shape
        # 平面中心点坐标与高度分离
        self.center_xy = (center[0], center[1])
        self.height = center[2]
        self.size = size
        # 二维几何形状用于平面相交检测
        self.geometry = self._create_geometry()

    def _create_geometry(self):
        x, y = self.center_xy
        if self.shape == 'circle':
            radius = self.size[0]
            return Point(x, y).buffer(radius)
        elif self.shape == 'rectangle':
            w, h = self.size
            return box(x - w/2, y - h/2, x + w/2, y + h/2)
        else:
            raise ValueError(f"Unsupported shape '{self.shape}'")

    def intersects_line(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> bool:
        """判断二维线段是否与障碍物相交"""
        line2d = LineString([point1, point2])
        return self.geometry.intersects(line2d)

    def blocks_flight(self,
                      p1: Tuple[float, float, float],
                      p2: Tuple[float, float, float]) -> bool:
        """判断三维飞行路径是否被障碍物阻挡"""
        line2d = LineString([union_xy(p1), union_xy(p2)])
        # 若二维平面投影相交，则根据高度判断
        if not self.geometry.intersects(line2d):
            return False
        return min(p1[2], p2[2]) <= self.height


class ObstacleMap:
    def __init__(self):
        """
        args_env: 包含 n_obst 的配置对象
        Map: 提供 ranges_x, ranges_y 的地图类
        """
        self.n_obst = args_env.n_obst
        self.map = Map()
        self.width_x = self.map.ranges_x
        self.width_y = self.map.ranges_y
        self.height_min = args_env.obst_height_min
        self.height_max = args_env.obst_height_max

        self.size_circle_min = args_env.size_circle_min
        self.size_circle_max = args_env.size_circle_max
        self.size_rectangle_min = args_env.size_rectangle_min
        self.size_rectangle_max = args_env.size_rectangle_max

        self.obstacles: List[Obstacle] = []
        self._generate_random_obstacles(self.n_obst)

    def _generate_random_obstacles(self, n: int, max_trials: int = 1000):
        trials = 0
        while len(self.obstacles) < n and trials < max_trials:
            trials += 1
            shape = random.choice(['circle', 'rectangle'])
            x = random.uniform(0, self.width_x)
            y = random.uniform(0, self.width_y)
            z = random.uniform(self.height_min, self.height_max)
            if shape == 'circle':
                size = (random.uniform(self.size_circle_min, self.size_circle_max), 0)
            else:
                size = (random.uniform(self.size_rectangle_min, self.size_rectangle_max), random.uniform(self.size_rectangle_min, self.size_rectangle_max))
            candidate = Obstacle(shape, (x, y, z), size)
            # 平面不重叠则加入
            if not any(obs.geometry.intersects(candidate.geometry) for obs in self.obstacles):
                self.obstacles.append(candidate)
        if len(self.obstacles) < n:
            print(f"警告：仅生成了 {len(self.obstacles)} 个非重叠障碍物（目标 {n} 个），可能空间不足。")

    def is_path_blocked_3d(self,
                           p1: Tuple[float, float, float],
                           p2: Tuple[float, float, float]) -> bool:
        """判断三维路径是否被任意障碍物阻挡"""
        return any(obs.blocks_flight(p1, p2) for obs in self.obstacles)


