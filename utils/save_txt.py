import os.path


class TXT_FILE(object):
    def __init__(self):
        self.path_pro = '/Users/tengman/Documents/Python/UAVTrajectory/pythonProject1103/results_data/'

    # 保存UAV、Job、GT特征
    def save_pro_txt(self, obj, label, time):
        # 保存Jobs属性
        if label == 0:
            file_name = os.path.join(self.path_pro, 'jobs.txt')
            # 清空文件内容
            with open(file_name, 'w') as file_obj:
                pass  # 仅仅打开并关闭文件，以清空文件内容
            for j in range(len(obj)):
                # 写入txt文件
                with open(file_name, 'a') as file_obj:
                    file_obj.write("job_id: {:d}, task number: {:d}, computing resources: {}, "
                                   "data size: {}, delay limited: {:d}\n".format(
                        obj[j].id, obj[j].n_tas, obj[j].s_comp, obj[j].s_data, obj[j].delay_limited))
        # 保存UAVs属性
        elif label == 1:
            file_name = os.path.join(self.path_pro, 'uav.txt')
            for u in range(len(obj)):
                with open(file_name, 'a') as file_obj:
                    file_obj.write("Time: {:d}, "
                                   "UAV_id: {:d}, "
                                   "Position: ({:d},{:d},{:d}), "
                                   "Velocities: {:d}, "
                                   "Transmission Power: {:d}, "
                                   "Computing Power: {:d}, "
                                   "Queue: {:f}\n".format(
                        time, obj[u].id, int(obj[u].pos[0]), int(obj[u].pos[1]), int(obj[u].pos[2]),
                        obj[u].vels, obj[u].p_tx, obj[u].p_cm, obj[u].queue))
            with open(file_name, 'a') as file_obj:
                file_obj.write("\n")

        # 保存GTs属性
        elif label == 2:
            file_name = os.path.join(self.path_pro, 'gts.txt')
            for g in range(len(obj)):
                if obj[g].re_job is None:
                    with open(file_name, 'a') as file_obj:
                        file_obj.write("Time: {:d}, "
                                       "GT_id: {:d}, "
                                       "Position: ({:d},{:d},{:d}), "
                                       "Label: {:d}, "
                                       "Requested Job_id: {}, "
                                       "Velocities: {:d}, "
                                       "Transmission Power: {:d}, "
                                       "Computing Power: {:d}, "
                                       "Queue: {:f}\n".format(
                            time, obj[g].id, int(obj[g].pos[0]), int(obj[g].pos[1]), int(obj[g].pos[2]),
                            obj[g].lab, obj[g].re_job, obj[g].vels, obj[g].p_tx, obj[g].p_cm, obj[g].queue))
                else:
                    with open(file_name, 'a') as file_obj:
                        file_obj.write("Time: {:d}, "
                                       "GT_id: {:d}, "
                                       "Position: ({:d},{:d},{:d}), "
                                       "Label: {:d}, "
                                       "Requested Job_id: {:d}, "
                                       "Velocities: {:d}, "
                                       "Transmission Power: {:d}, "
                                       "Computing Power: {:d}, "
                                       "Queue: {:f}\n".format(
                            time, obj[g].id, int(obj[g].pos[0]), int(obj[g].pos[1]), int(obj[g].pos[2]),
                            obj[g].lab, obj[g].re_job.id, obj[g].vels, obj[g].p_tx, obj[g].p_cm, obj[g].queue))

            with open(file_name, 'a') as file_obj:
                    file_obj.write("\n")

    # 清空UAV特征文件
    def clear_uav_gts(self):
        file_name_uav = os.path.join(self.path_pro, 'uav.txt')
        file_name_gts = os.path.join(self.path_pro, 'gts.txt')
        # 清空文件内容
        with open(file_name_uav, 'w') as file_obj_uav:
            pass  # 仅仅打开并关闭文件，以清空文件内容
        # 清空文件内容
        with open(file_name_gts, 'w') as file_obj_gts:
            pass  # 仅仅打开并关闭文件，以清空文件内容

    # 保存task特征
    def save_task_pro_txt(self, uav_id, job_id, task_id, start_time, action, fini_time, reward, trans_time, comp_time, wait_time):
        file_name = os.path.join(self.path_pro, 'offload.txt')
        with open(file_name, 'a') as file_obj:
            file_obj.write("UAV_id: {:d}, "
                           "Job_id: {:d}, "
                           "task_id: {:d}, "
                           "start_time: {:.2f}, "
                           "actions: {:d}, "
                           "fini_time: {:.2f}, "
                           "reward: {:.2f},   "
                           "trans_time: {:.2f}, "
                           "comp_time: {:.2f}, "
                           "wait_time: {:.2f}\n".format(
                uav_id, job_id, task_id, start_time, action, fini_time, reward, trans_time, comp_time, wait_time))

    # 清楚Task特征
    def clear_task_off(self):
        file_name = os.path.join(self.path_pro, 'offload.txt')
        with open(file_name, 'w') as file_obj:
            pass  # 仅仅打开并关闭文件，以清空文件内容

    def gap_off_txt(self):
        file_name = os.path.join(self.path_pro, 'offload.txt')
        with open(file_name, 'a') as file_obj:
            file_obj.write("\n")

    def save_rewards_txt(self, action, rewards, reward_time, reward_energy, reward_1, reward_2, reward_ave_time):
        file_name = os.path.join(self.path_pro, 'rewards.txt')
        with open(file_name, 'a') as file_obj:
            file_obj.write("actions: {:d}, "
                           "rewards: {:f}, "
                           "reward_time: {:f}, "
                           "reward_energy: {:f}, "
                           "reward_1: {:f}, "
                           "reward_2: {:f},"
                           "reward_ave_time: {:f} \n".format(action, rewards, reward_time, reward_energy, reward_1, reward_2, reward_ave_time))

    def clear_rewards_txt(self):
        file_name = os.path.join(self.path_pro, 'rewards.txt')
        with open(file_name, 'w') as file_obj:
            pass  # 仅仅打开并关闭文件，以清空文件内容

    def save_state_txt(self, state):
        file_name = os.path.join(self.path_pro, 'states.txt')

        # 当前task属性
        task = state['task']
        obs_1 = [task.job_id, task.s_data, task.s_comp, task.req_uav, task.task_id, task.req_time, task.remain_time]
        with open(file_name, 'a') as file_obj:
            file_obj.write(" ".join(map(str, obs_1)) + " ")

        # 当前时间和当前任务数量
        obs_2 = [state['time'], state['task_num']]
        with open(file_name, 'a') as file_obj:
            file_obj.write(" ".join(map(str, obs_2)) + " ")

        # uav剩余电量属性
        uav_feats = state['uav']
        obs_3 = []
        for u in range(len(uav_feats)):
            obs_3.append(uav_feats[u][5])
        with open(file_name, 'a') as file_obj:
            file_obj.write(" ".join(map(str, obs_3)) + " ")

        # uav当前队列属性
        obs_4 = []
        for u in range(len(uav_feats)):
            obs_4.append(uav_feats[u][6])
        with open(file_name, 'a') as file_obj:
            file_obj.write(" ".join(map(str, obs_4)) + " ")

        # uav位置
        obs_5 = []
        for u in range(len(uav_feats)):
            obs_5.append(uav_feats[u][0])
            obs_5.append(uav_feats[u][1])
            obs_5.append(uav_feats[u][2])
        with open(file_name, 'a') as file_obj:
            file_obj.write(" ".join(map(str, obs_5)) + " ")

        with open(file_name, 'a') as file_obj:
            file_obj.write("\n")

    def clear_state_txt(self):
        file_name = os.path.join(self.path_pro, 'states.txt')
        with open(file_name, 'w') as file_obj:
            pass  # 仅仅打开并关闭文件，以清空文件内容

    def save_state_normal_txt(self, state):
        file_name = os.path.join(self.path_pro, 'states_normal.txt')

        with open(file_name, 'a') as file_obj:
            file_obj.write(" ".join(map(str, state)) + "\n")

    def clear_state_normal_txt(self):
        file_name = os.path.join(self.path_pro, 'states_normal.txt')
        with open(file_name, 'w') as file_obj:
            pass  # 仅仅打开并关闭文件，以清空文件内容


class save_result(object):
    def __init__(self, log_dir_result):
        self.txt_name = f"rewards_evaluate_avg.txt"
        self.path_pro = log_dir_result

    # def save_reward_txt(self, algorithm, n_uav, n_gts, reward_avg, reward_time, reward_load, reward_success, reward_energy, rate):
    #     file_name = os.path.join(self.path_pro, self.txt_name)
    #     with open(file_name, 'a') as file_obj:
    #         file_obj.write("alg: {}, "
    #                        "n_uav: {:d}, "
    #                        "n_gts: {:d}, "
    #                        "reward_avg: {:.2f}, "
    #                        "reward_time: {:.2f}, "
    #                        "reward_load: {:.2f}, "
    #                        "reward_success: {:.2f}, "
    #                        "reward_energy: {:.2f}, "
    #                        "reward_rate: {:.2f}\n".format(
    #             algorithm, n_uav, n_gts, reward_avg.item(), reward_time.item(), reward_load.item(), reward_success.item(), reward_energy.item(), rate.item()))

    def save_reward_txt(self, algorithm, n_uav, n_gts, reward_avg):
        file_name = os.path.join(self.path_pro, self.txt_name)
        with open(file_name, 'a') as file_obj:
            file_obj.write("alg: {}, "
                           "n_uav: {:d}, "
                           "n_gts: {:d}, "
                           "reward_avg: {:.2f}\n".format(
                algorithm, n_uav, n_gts, reward_avg.item()))

    def clear_reward_txt(self):
        file_name = os.path.join(self.path_pro, self.txt_name)
        with open(file_name, 'w') as file_obj:
            pass  # 仅仅打开并关闭文件，以清空文件内容
