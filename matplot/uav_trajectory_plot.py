import re
import matplotlib.pyplot as plt

def plot_uav_and_jobs(uav_file, job_file, save_path=None):
    """
    绘制 UAV 轨迹 + Job 位置图
    :param uav_file: UAV 轨迹 txt 文件路径
    :param job_file: Job 位置 txt 文件路径
    :param save_path: 保存路径（如 "output.png"），不需要保存则为 None
    """
    uav_trajs = {}

    # === Load UAV trajectory ===
    with open(uav_file, "r") as f:
        for line in f:
            # 提取 uav_id 和 uav_pos
            m = re.search(r"uav_id:(\d+).*uav_pos:\[(.*?)\]", line)
            if not m:
                continue

            uav_id_str, pos_str = m.groups()
            uav_id = int(uav_id_str)

            # 解析坐标
            coords = re.split(r"[,\s]+", pos_str.strip())
            coords = [c for c in coords if c != ""]
            if len(coords) < 2:
                continue

            x = float(coords[0])
            y = float(coords[1])

            # 以 uav_id 为 key 存轨迹
            if uav_id not in uav_trajs:
                uav_trajs[uav_id] = {"x": [], "y": []}
            uav_trajs[uav_id]["x"].append(x)
            uav_trajs[uav_id]["y"].append(y)

    # === Load Job positions ===
    job_x, job_y = [], []

    with open(job_file, "r") as f:
        for line in f:
            m = re.search(r"job_pos:\s*\[(.*?)\]", line)  # 允许空格
            if not m:
                continue

            pos_str = m.group(1)
            coords = re.split(r"[,\s]+", pos_str.strip())
            coords = [c for c in coords if c != ""]
            if len(coords) < 2:
                continue

            job_x.append(float(coords[0]))
            job_y.append(float(coords[1]))

    # === Plot ===
    plt.figure(figsize=(8, 6))

    # UAV tracks: small circles
    for lab, traj in sorted(uav_trajs.items()):
        plt.plot(traj["x"], traj["y"], marker='o', markersize=3,
                 linewidth=1.2, label=f"UAV {lab}")

    # Jobs: red triangles
    plt.scatter(job_x, job_y, marker='^', color='red', s=40, label="Jobs")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("UAV Trajectories and Job Positions")
    # plt.legend()
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=220, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

