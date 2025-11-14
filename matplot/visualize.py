from typing import List

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from envs.obstacle import ObstacleMap, Obstacle
from shapely.geometry import Polygon

import matplotlib

# 设置支持中文字体，防止缺失 glyph 报错
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False


def visualize_obstacles(obstacles: List[Obstacle], width: float, height: float):
    """
    在二维平面上绘制障碍物分布
    obstacles: 障碍物列表
    width, height: 地图宽高
    """
    fig, ax = plt.subplots()
    for obs in obstacles:
        x, y = obs.center_xy
        if obs.shape == 'circle':
            circle = Circle((x, y), obs.size[0], fill=False, alpha=0.5)
            ax.add_patch(circle)
        else:
            w, h = obs.size
            rect = Rectangle((x - w/2, y - h/2), w, h, fill=False, alpha=0.5)
            ax.add_patch(rect)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('障碍物平面分布可视化')
    plt.show()


# 示例用法（若作为独立脚本运行）
if __name__ == '__main__':
    obs_map = ObstacleMap()
    polygons: List[Polygon] = [obs.geometry for obs in obs_map.obstacles]
    print(polygons)

    p1 = (0, 0, 0)
    p2 = (300, 300, 0)
    is_blocked = obs_map.is_path_blocked_3d(p1, p2)
    print(is_blocked)

    visualize_obstacles(obs_map.obstacles, obs_map.width_x, obs_map.width_y)
