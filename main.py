import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
from server import *
from user import *
from load_save import *
from plots import *





def main1():
  mu_list = [0.9, 0.8 ,0.7, 0.6, 0.5 ,0.4, 0.3, 0.2 ,0.1]

  K = len(mu_list)
  N = 6
  c = 0.1
  p_0 = 0.6
  alpha = 0.5
  beta = 0.8
  d = 0.05

  time_end = int(3e4)
  simulation_time = 10
  ############ check ##########
  A = np.pad(np.sort(mu_list), (0, 1))
  B = np.pad(np.sort(mu_list), (1, 0))
  min_distance_between_rewards = np.min((A-B)[:-1])
  if min_distance_between_rewards < d:
    warnings.warn("This is a warning message.")

  mega_args = {"p_0": p_0, "alpha": alpha, "beta": beta, "c": c, "d": d, "K": K}
  num_users_on_channels_MEGA, users_MEGA = run_simulation(user_MEGA, N, mu_list, time_end, mega_args)
  rand_args = {"K": K}
  num_users_on_channels_RAND, users_RAND = run_simulation(user_RAND, N, mu_list, time_end, rand_args)

  # plot collision
  collision_over_time_MEGA = compute_collision(num_users_on_channels_MEGA, N)
  collision_over_time_RAND = compute_collision(num_users_on_channels_RAND, N)
  
  plot_dict("collision_over_time", {'MEGA - d < Delta':collision_over_time_MEGA, 'RAND':collision_over_time_RAND})

  # extract reward per user, then print it
  reward_mat_mega = np.array([user.reward for user in users_MEGA], dtype=np.int8).T
  reward_mat_rand = np.array([user.reward for user in users_RAND], dtype=np.int8).T

  regret_MEGA = compute_regret(mu_list, reward_mat_mega)
  regret_RAND = compute_regret(mu_list, reward_mat_rand)
  
  plot_dict("regret_over_time", {'MEGA - d < Delta':regret_MEGA, 'RAND':regret_RAND})
  
  # save
  save_matrix_with_mu(num_users_on_channels_MEGA, mu_list, 'MEGA_num_users_on_channel')
  save_matrix_with_mu(num_users_on_channels_RAND, mu_list, 'RAND_num_users_on_channel')

  user_idx_list = list(range(N))
  save_matrix_with_mu(reward_mat_mega, user_idx_list, 'MEGA_reward_per_user')
  save_matrix_with_mu(reward_mat_rand, user_idx_list, 'RAND_reward_per_user')


if __name__ == "__main__":
  main1()