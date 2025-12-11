import numpy as np
from sklearn.cluster import KMeans


def compute_hotspots_from_jobs(jobs, n_hotspots: int, random_state: int = 0):
    """
    根据 Job 对象列表计算热点位置。

    参数
    ----
    jobs : list[Job]
        每个 job 需要有 .pos 属性，job.pos[0], job.pos[1] 是 x, y 坐标。
    n_hotspots : int
        指定热点个数，例如 1 / 2 / 3。
    random_state : int
        KMeans 的随机种子，方便复现。

    返回
    ----
    hotspots : np.ndarray, shape = (K, 2)
        计算得到的 K 个热点坐标（K = min(n_hotspots, len(jobs))）
    """
    assert len(jobs) > 0, "jobs 不能为空"
    assert n_hotspots >= 1, "n_hotspots 必须 >= 1"

    # 把 job.pos 抽取成 (N_job, 2) 的数组，只取 x, y
    job_pos = np.array([[float(j.pos[0]), float(j.pos[1])] for j in jobs], dtype=float)
    n_jobs = job_pos.shape[0]
    K = min(n_hotspots, n_jobs)

    # 只有 1 个热点：所有任务的几何中心
    if K == 1:
        hotspot = job_pos.mean(axis=0, keepdims=True)  # shape = (1, 2)
        return hotspot

    # 多个热点：KMeans 聚类
    kmeans = KMeans(n_clusters=K, random_state=random_state, n_init="auto")
    kmeans.fit(job_pos)
    hotspots = kmeans.cluster_centers_  # shape = (K, 2)
    return hotspots


def assign_hotspots_to_uav(uavs, hotspots: np.ndarray):
    """
    为每个 UAV（对象）选择最近的热点。

    参数
    ----
    uavs : list[UAV]
        每个 uav 需要有 .pos 属性，uav.pos[0], uav.pos[1] 是 x, y 坐标。
    hotspots : np.ndarray, shape = (K, 2)
        K 个热点坐标（由 compute_hotspots_from_jobs 得到）。

    返回
    ----
    assigned_idx : np.ndarray, shape = (N_uav,)
        每个 UAV 对应的热点索引（0 ~ K-1）。
    assigned_hotspots : np.ndarray, shape = (N_uav, 2)
        每个 UAV 对应的热点坐标。
    """
    hotspots = np.asarray(hotspots, dtype=float)
    assert hotspots.ndim == 2 and hotspots.shape[1] == 2, "hotspots 必须是 (K, 2)"

    # 抽取 UAV 的 (x, y)
    uav_pos = np.array([[float(u.pos[0]), float(u.pos[1])] for u in uavs], dtype=float)
    n_uav = uav_pos.shape[0]

    if n_uav == 0:
        return np.array([], dtype=int), np.empty((0, 2), dtype=float)

    # 计算每个 UAV 到每个热点的距离: (N_uav, K)
    dists = np.linalg.norm(uav_pos[:, None, :] - hotspots[None, :, :], axis=2)

    # 每个 UAV 选择距离最小的那个热点
    assigned_idx = np.argmin(dists, axis=1)
    assigned_hotspots = hotspots[assigned_idx]

    return assigned_idx, assigned_hotspots


# # 假设环境
# ranges_1 = [300, 300]
# args_env.ranges_x = ranges_1[0]
# args_env.ranges_y = ranges_1[1]
# args_env.n_uav = 9
# args_env.n_jobs = 50
# env = Environment_1()
# env.reset()
# # 1) 选择热点个数，比如 1 / 2 / 3
# n_hotspots = 1
#
# # 2) 计算热点位置
# hotspots = compute_hotspots_from_jobs(env.jobs, n_hotspots)
#
# # 3) 为每个 UAV 分配最近的热点
# assigned_idx, assigned_hotspots = assign_hotspots_to_uav(env.uav, hotspots)
#
# print("Hotspots:\n", hotspots)
# print("每个 UAV 对应的热点索引:", assigned_idx)
# print("每个 UAV 对应的热点坐标:\n", assigned_hotspots)

