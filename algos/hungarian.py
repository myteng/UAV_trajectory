from typing import List, Tuple
import numpy as np


def hungarian_min_cost(cost: List[List[float]]) -> Tuple[List[int], float]:
    """
    匈牙利算法（Kuhn–Munkres），求方阵成本的最小权完美匹配。
    输入:
        cost: n x n 矩阵，cost[i][j] 为任务 i 在 UAV j 上的成本
    输出:
        assignment: 长度 n，assignment[i] = j
        total_cost: 匹配的总成本
    复杂度: O(n^3)
    """
    n = len(cost)
    # 1) 行/列势（potentials）
    u = [0.0] * (n + 1)  # 行势，1..n 用，0 做虚列
    v = [0.0] * (n + 1)  # 列势，1..n 用，0 做虚列
    # 2) p[j]：列 j 的匹配行；way[j]：最短路上一跳前驱列
    p = [0] * (n + 1)
    way = [0] * (n + 1)

    for i in range(1, n + 1):  # 逐行增广
        p[0] = i
        minv = [float('inf')] * (n + 1)  # 到各列的“剩余代价”最短距离
        used = [False] * (n + 1)
        j0 = 0  # 当前“活动列”从虚列 0 开始
        while True:
            used[j0] = True
            i0 = p[j0]  # 当前活动列匹配到的行
            delta = float('inf')
            j1 = 0
            # 3) 以“紧边度量”松弛：cur = cost[i0][j] - u[i0] - v[j]
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            # —— 在这里插入守卫：如果本轮没有任何未访问列被有效松弛，delta 会保持为 inf ——
            if not (delta < float('inf')):
                raise ValueError(
                    "Hungarian: no feasible augmenting step. "
                    "Cost matrix likely contains Inf/NaN or all remaining edges are prohibited. "
                    "Use a large finite big_M instead of np.inf, and ensure padding is correct."
                )
            # 4) 调整势能 u,v：把所有已用列对应的行势 +delta，未用列的 minv 减 delta
            for j in range(0, n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1  # 转到新的最优列
            # 5) 如果该列还没匹配，找到一条增广路径，退出循环
            if p[j0] == 0:
                break
        # 6) 回溯 way[]，沿路径翻转匹配，完成一次增广
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    # 7) p[j] = i（列到行），反解为 assignment[i-1] = j-1
    assignment = [-1] * n
    for j in range(1, n + 1):
        i = p[j]
        assignment[i - 1] = j - 1
    total_cost = sum(cost[i][assignment[i]] for i in range(n))
    return assignment, total_cost


def assign_tasks(C_sub, big_M=1e6):
    n_tasks, n_uav = C_sub.shape
    n = max(n_tasks, n_uav)

    # 1）初始化补齐矩阵
    padded = np.full((n, n), big_M)
    padded[:n_tasks, :n_uav] = C_sub

    # 2）若任务数 < UAV 数，虚拟任务代价应为 0
    if n_tasks < n_uav:
        padded[n_tasks:, :n_uav] = 0.0

    # —— 这里加消毒（非常关键，避免 delta==inf 的源头）——
    padded = np.asarray(padded, dtype=np.float64)
    padded = np.nan_to_num(padded, nan=big_M, posinf=big_M, neginf=big_M)
    padded[padded > big_M] = big_M

    # 3）调用匈牙利算法
    assignment_square, _total_cost_padded = hungarian_min_cost(padded)

    # 4) 过滤真实匹配（去掉匹配到虚拟行/列的项）
    real_assignment = []
    for i in range(min(n_tasks, n)):  # 只遍历真实任务行
        j = assignment_square[i]
        if j < n_uav and not np.isinf(C_sub[i][j]) and not np.isnan(C_sub[i][j]):  # 只保留真实UAV列
            real_assignment.append((i, j))

    # 5) 计算真实目标值（注意：不要用 padded 的 total）
    real_cost = np.round(sum(C_sub[i, j] for i, j in real_assignment), 2)
    return real_assignment, real_cost


def assign_task_with_submatrix(cost_matrix, big_M=1e6):
    """
        输入:
          cost_matrix: shape=(n_jobs, n_uavs)，允许含 np.inf
        过程:
          1) 剔除整行为 inf 的 job；剔除整列为 inf 的 uav；
          2) 只对保留下来的子矩阵做一次分配；
          3) 将结果映射回原始 job_id / uav_id；
          4) 返回与原矩阵同尺寸的一热分配矩阵（行列与 id 对齐）。
        返回:
          pairs: [(job_id, uav_id), ...]   # 使用原始索引
          assign_full: (n_jobs, n_uavs) 一热矩阵
          total_cost: 在原 cost_matrix 上的真实成本总和（忽略 inf）
          kept_job_idx: 子矩阵行对应的原始 job 索引
          kept_uav_idx: 子矩阵列对应的原始 uav 索引
    """
    C = np.array(cost_matrix, dtype=float)
    n_jobs, n_uav = C.shape

    # 1) 构造掩码：保留“非全 inf 的行/列”
    row_mask = ~np.isinf(C).all(axis=1)  # True 表示该 job 行可参与
    col_mask = ~np.isinf(C).all(axis=0)  # True 表示该 uav 列可参与

    kept_job_idx = np.where(row_mask)[0]
    kept_uav_idx = np.where(col_mask)[0]

    # 若任一侧为空，直接返回空解
    if kept_job_idx.size == 0 or kept_uav_idx.size == 0:
        return [], np.zeros_like(C, dtype=np.float32), 0.0, kept_job_idx, kept_uav_idx

    # 2) 子矩阵
    C_sub = C[np.ix_(kept_job_idx, kept_uav_idx)]

    # 3) 子矩阵内做长方阵→方阵匈牙利
    pairs_sub, real_cost = assign_tasks(C_sub, big_M=big_M)

    # 4) 回映射回原始 id
    pairs = [(int(kept_job_idx[i]), int(kept_uav_idx[j])) for (i, j) in pairs_sub]

    # 5) 构造与原矩阵同尺寸的一热分配矩阵
    assign_full = np.zeros((n_jobs, n_uav), dtype=np.float32)
    for i, j in pairs:
        assign_full[i, j] = 1.0

    # —— 插入守卫：如果本轮卸载的成本cost_value = inf，则报错 ——
    if not np.isfinite(real_cost).any():
        raise ValueError(
            "Cost matrix likely contains Inf/NaN or all remaining edges are prohibited. "
        )

    return assign_full, real_cost


# ------------- 示例 -------------
if __name__ == "__main__":
    C = [
        [9, 11, 14],
        [6, 15, 13],
        [12, 13, 6],
    ]
    assignment, total_cost = hungarian_min_cost(C)
    print("最优分配(任务 -> UAV):", assignment)
    print("最小总成本:", total_cost)

    # 示例
    C = [
        [9, 11, 14, 19],
        [6, 15, 13, 11],
        [12, 13, 6, 10],
    ]
    assign = assign_tasks(C)
    print("任务-UAV分配:", assign)
