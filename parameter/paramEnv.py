import argparse

import numpy as np

parser = argparse.ArgumentParser(description='ENV_NODEs')

# # - Seed -
# parser.add_argument('--seed', type=int, default=0,
#                     help='The random seed (default: 0)')

# -- Environment --
parser.add_argument('--use_potential_reward', type=bool, default=True)

# - Range -
parser.add_argument('--n_grids', type=int, default=9,
                    help='The number of grids (default: 200m)')
parser.add_argument('--ranges_x', type=int, default=500,
                    help='The X-axis range coverage (default: 300m)')
parser.add_argument('--ranges_y', type=int, default=500,
                    help='The Y-axis range coverage (default: 300m)')
parser.add_argument('--range_pos', type=int, default=100,
                    help='The range coverage for each UAV (default: 100*100)')
parser.add_argument('--h_uav', type=int, default=100,
                    help='The Height for each UAV (default: 100m)')
parser.add_argument('--r_safe', type=int, default=10,
                    help='The safe range for each UAV (default: 10)')

parser.add_argument('--episode_limit', type=int, default=20,
                    help='Maximum number of timesteps in an episode (default: 20)')
parser.add_argument('--spt', type=int, default=5,
                    help='Time length of each step (sec) (default: 5)')
parser.add_argument('--r_comm', type=float, default=np.inf,
                    help='Range of multi-agent communication (m) (default: np.inf)')

# - UAV -
parser.add_argument('--n_uav', type=int, default=9,
                    help='The Number of UAVs (default: 9)')
# - Energy -
parser.add_argument('--max_energy_uav', type=int, default=500,
                    help='The maximum Energy of UAV (default: 100)')
# - Velocities -
parser.add_argument('--vels_uav', type=int, default=10,
                    help='The Velocities of UAVs (m/s) (default: 10)')
# - Transmission Power (dbm) -
parser.add_argument('--p_tx_uav_min', type=int, default=10,
                    help='The min transmission power of UAVs (default: 10dbm)')
parser.add_argument('--p_tx_uav_max', type=int, default=20,
                    help='The max transmission power of UAVs (default: 20dbm)')
parser.add_argument('--max_distance', type=int, default=150,
                    help='The max transmission distance between UAVs (default: 150m)')
# - Computing Power（MHz）-
parser.add_argument('--p_cm_uav_min', type=int, default=50,
                    help='The min computing power of UAVs（MHz） (default: 50)')
parser.add_argument('--p_cm_uav_max', type=int, default=100,
                    help='The max computing power of UAVs（MHz） (default: 100)')
# Energy Consumption Coefficient
parser.add_argument('--fly_energy_coef', type=float, default=1.0,
                    help='The fly energy consumption coefficient of UAVs（） (default: 1.0)')
parser.add_argument('--hov_energy_coef', type=float, default=0.1,
                    help='The hovering energy consumption coefficient of UAVs（） (default: 1.0)')
parser.add_argument('--comp_energy_coef', type=float, default=1e-6,
                    help='The computing energy consumption coefficient of UAVs（） (default: 1.0)')
parser.add_argument('--send_energy_coef', type=float, default=0.1,
                    help='The send energy consumption coefficient of UAVs（） (default: 1.0)')

parser.add_argument('--max_parallel_tasks', type=int, default=5,
                    help='The maximum number of parallel processing tasks for UAVs（） (default: 5)')

# # - Direction -
# parser.add_argument('--dir', type=list, default=[-2, -1, 0, 1, 2],
#                     help='The directions of GTs/UAVs (default: [-2, -1, 0, 1, 2])')


# # - GT -
# parser.add_argument('--n_gts', type=int, default=9,
#                     help='The Number of GTs (default: 9)')
# # - Queue -
# parser.add_argument('--max_queue_gts', type=int, default=5,
#                     help='The maximum task queue of GTs (default: 5)')
# # - Velocities -
# parser.add_argument('--vels_gts', type=int, default=5,
#                     help='The Velocities of GTs (m/s) (default: 5)')
# # - Transmission Power -
# parser.add_argument('--p_tx_gts', type=int, default=5,
#                     help='The transmission power of GTs (default: 5dbm)')
# # - Computing Power（MHz）-
# parser.add_argument('--p_cm_gts', type=int, default=50,
#                     help='The computing power of GTs（MHz） (default: 50)')


# - Jobs -
parser.add_argument('--n_jobs', type=int, default=100,
                    help='The Number of Jobs (default: 10)')
parser.add_argument('--job_arr_time', type=int, default=0,
                    help='The arrival time of Jobs (default: 0)')
# parser.add_argument('--n_tas_max', type=int, default=4,
#                     help='The Maximum Number of Task for each Job (default: 4)')

parser.add_argument('--workload_min', type=int, default=200,
                    help='The Minimum Computing Resource Required by Jobs (Megacycle) (default: 200)')
parser.add_argument('--workload_max', type=int, default=500,
                    help='The Maximum Computing Resource Required by Jobs (Megacycle) (default: 500)')
parser.add_argument('--data_size_min', type=int, default=500,
                    help='The Minimum Data Size between Tasks (KB) (default: 500)')
parser.add_argument('--data_size_max', type=int, default=1000,
                    help='The Maximum Data Size between Tasks (KB) (default: 1000)')
parser.add_argument('--deadline_min', type=int, default=300,
                    help='The Minimum Limited Delay Time (s) (default: 10)')
parser.add_argument('--deadline_max', type=int, default=500,
                    help='The Maximum Limited Delay Time (s) (default: 40)')
parser.add_argument('--n_task_min', type=int, default=1,
                    help='The Minimum Number of Tasks (default: 1)')
parser.add_argument('--n_task_max', type=int, default=5,
                    help='The Maximum Number of Tasks (default: 5)')
parser.add_argument('--value_min', type=int, default=1,
                    help='The Minimum Value (default: 1)')
parser.add_argument('--value_max', type=int, default=5,
                    help='The Maximum Value (default: 5)')
#
# parser.add_argument('--n_tas', type=list, default=[1, 2, 3],
#                     help='The Number of Task for each Job (default: 4)')
# parser.add_argument('--s_comp', type=list, default=[100, 150, 200],
#                     help='The Computing Resource Required by Jobs (Megacycle) (default: 150)')
# parser.add_argument('--s_data', type=list, default=[200, 500, 800],
#                     help='The Data Size between Tasks (KB) (default: 1000)')
# parser.add_argument('--delay_limit', type=list, default=[5, 10, 15],
#                     help='The Maximum Limited Delay Time (s) (default: 3)')


# - Obstacles -
parser.add_argument('--n_obst', type=int, default=10,
                    help='The Number of Obstacles (default: 10)')
parser.add_argument('--obst_height_min', type=int, default=5,
                    help='The Minimum Height of Obstacles (default: 5)')
parser.add_argument('--obst_height_max', type=int, default=20,
                    help='The Maximum Height of Obstacles (default: 20)')
parser.add_argument('--size_circle_min', type=int, default=3,
                    help='The Minimum Size of Circle Obstacles (default: 3)')
parser.add_argument('--size_circle_max', type=int, default=8,
                    help='The Maximum Size of Circle Obstacles (default: 8)')
parser.add_argument('--size_rectangle_min', type=int, default=5,
                    help='The Minimum Size of Rectangle Obstacles (default: 5)')
parser.add_argument('--size_rectangle_max', type=int, default=10,
                    help='The Maximum Size of Rectangle Obstacles (default: 10)')

# - Channel -
parser.add_argument('--bw', type=int, default=180e3,
                    help='The Bandwidth of sub-channels (Hz) (default: 180e3)')
parser.add_argument('--fc', type=int, default=2.4e9,
                    help='The Central carrier frequency (Hz) (default: 2.4e9)')
parser.add_argument('--no', type=int, default=-170,
                    help='The noise power (dbm) (default: -170)')
parser.add_argument('--scene', type=str, default='dense-urban',
                    help='The Scene of channel model (default: dense-urban)')
parser.add_argument('--datasets', type=bool, default=False,
                    help='Use the building dataset (default: False)')
parser.add_argument('--p_tx_gts', type=int, default=5,
                    help='The transmission power of GTs (default: 5dbm)')

args_env = parser.parse_args()