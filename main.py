import numpy as np
import warnings
from server import *
from user import *
from load_save import *
from plots import *
from mu_change import *
import random

def channel_prodication_error(channel_order:np.ndarray, time_end:int, K:int) -> np.ndarray:
  unique_channel_selection_error = np.zeros([time_end, K])
  for t in range(time_end):
    for k in range(K):
      unique_channel_selection_error[t, k] = np.unique(channel_order[:,t,k]).size - 1
  total_error = np.sum(unique_channel_selection_error, axis= 1)
  return total_error

def main1(seed = 0):
  # mu_list = [0.9, 0.8 ,0.7, 0.6, 0.5 ,0.4, 0.3, 0.2 ,0.1]
  mu_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
  # mu_list = [0.9, 0.72 ,0.71, 0.73, 0.7 ,0.69, 0.68, 0.2 ,0.1]
  # mu_list = [0.9, 0.89 ,0.88, 0.87, 0.86 ,0.85, 0.2, 0.2 ,0.2]
  # mu_list = [0.6, 0.55, 0.52, 0.51, 0.5, 0.49, 0.48, 0.47, 0.46, 0.45]
  # mu_list = rng.permutation(mu_list)
  mu_list = np.array(mu_list)

  K = len(mu_list)
  N = 6
  c = 0.1
  p_0 = 0.6
  alpha = 0.5
  beta = 0.8
  d = 0.05
  delta = 0.05
  avg_size = 200
  min_explore_value = 0.2

  time_end = int(1e4)
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
  mega_args['avg_size'] = avg_size
  mega_args['min_explore_value'] = min_explore_value
  num_users_on_channels_MEGA_cal, users_MEGA_cal = run_simulation(user_MEGA_col2, N, mu_list, time_end, mega_args)

  # plot collision
  collision_over_time_MEGA = compute_collision(num_users_on_channels_MEGA, N)
  collision_over_time_RAND = compute_collision(num_users_on_channels_RAND, N)
  collision_over_time_MEGA_cal = compute_collision(num_users_on_channels_MEGA_cal, N)
  
  plot_dict("collision_over_time", {'MEGA - d < Delta':collision_over_time_MEGA, 'RAND':collision_over_time_RAND, 'MEGA - cal' : collision_over_time_MEGA_cal})

  # extract reward per user, then print it
  reward_mat_mega = np.array([user.reward for user in users_MEGA], dtype=np.int8).T
  reward_mat_rand = np.array([user.reward for user in users_RAND], dtype=np.int8).T
  reward_mat_mega_cal = np.array([user.reward for user in users_MEGA_cal], dtype=np.int8).T

  regret_MEGA = compute_regret(mu_list, reward_mat_mega)
  regret_RAND = compute_regret(mu_list, reward_mat_rand)
  regret_MEGA_cal = compute_regret(mu_list, reward_mat_mega_cal)

  plot_dict("regret_over_time", {'MEGA - d < Delta':regret_MEGA, 'RAND':regret_RAND, 'MEGA - cal' : regret_MEGA_cal})
  
  plot_shape = [3,6]
  plot_subplots("num users on channels, MEGA", {i: num_users_on_channels_MEGA[:, i].reshape(-1, 100).mean(axis=1) for i in range(len(mu_list))}, plot_shape)
  plot_subplots("num users on channels, RAND", {i: num_users_on_channels_RAND[:, i].reshape(-1, 100).mean(axis=1) for i in range(len(mu_list))}, plot_shape)
  plot_subplots("num users on channels, MEGA_cal", {i: num_users_on_channels_MEGA_cal[:, i].reshape(-1, 100).mean(axis=1) for i in range(len(mu_list))}, plot_shape)

  # # save
  # save_matrix_with_mu(num_users_on_channels_MEGA, mu_list, 'MEGA_num_users_on_channel')
  # save_matrix_with_mu(num_users_on_channels_RAND, mu_list, 'RAND_num_users_on_channel')

  # user_idx_list = list(range(N))
  # save_matrix_with_mu(reward_mat_mega, user_idx_list, 'MEGA_reward_per_user')
  # save_matrix_with_mu(reward_mat_rand, user_idx_list, 'RAND_reward_per_user')

def main3(seed = 0):
  time_end = int(1e4)
  params = [
    {'power': 0.9, 'change': 0.05, 'w': 10000, 'theta': 0}, # V
    {'power': 0.7, 'change': 0.2, 'w': 10000, 'theta': np.pi / 2}, # V
    {'power': 0.6, 'change': 0.35, 'w': 30000, 'theta': np.pi / 2}, # V
    {'power': 0.7, 'change': 0.15, 'w': 10000, 'theta': 0}, # V
    {'power': 0.74, 'change': 0.1, 'w': 40000, 'theta': np.pi / 2},
    {'power': 0.8, 'change': 0.2, 'w': 30000, 'theta': -np.pi / 2}, # V
    {'power': 0.78, 'change': 0.05, 'w': 10000, 'theta': np.pi / 4},
    {'power': 0.7, 'change': 0.1, 'w': 15000, 'theta': np.pi / 8},
    {'power': 0.7, 'change': 0, 'w': 10000, 'theta': 0}, # V
  ]

  mu_list = generate_mu_change(time_end=time_end, params=params)
  time_change = 7e3
  params = [
    {'value':[0.9,0.3], 'time':[time_change]},
    {'value':[0.8,0.2], 'time':[time_change]},
    {'value':[0.7,0.1], 'time':[time_change]},
    {'value':[0.6], 'time':[]},
    {'value':[0.5], 'time':[]},
    {'value':[0.4], 'time':[]},
    {'value':[0.3,0.9], 'time':[time_change]},
    {'value':[0.2,0.8], 'time':[time_change]},
    {'value':[0.1,0.7], 'time':[time_change]},
    ]
  mu_list = generate_mu_constant(time_end=time_end, params=params)
  X, K = mu_list.shape
  plot_dict("plot_mu_list", {j: mu_list[:, j] for j in range(K)})
  perm = np.random.permutation(mu_list.shape[1])  # shuffled indices for columns
  mu_list = mu_list[:, perm]

  N = 6
  c = 0.1
  p_0 = 0.6
  alpha = 0.5
  beta = 0.8
  d = 0.05
  avg_size = 200
  min_explore_value = 0.1

  ############ check ##########
  mega_args = {"p_0": p_0, "alpha": alpha, "beta": beta, "c": c, "d": d, "K": K}
  num_users_on_channels_MEGA, users_MEGA = run_simulation_mu_changes(user_MEGA, N, mu_list, time_end, mega_args)
  rand_args = {"K": K}
  num_users_on_channels_RAND, users_RAND = run_simulation_mu_changes(user_RAND, N, mu_list, time_end, rand_args)
  mega_args['avg_size'] = avg_size
  mega_args['min_explore_value'] = min_explore_value
  num_users_on_channels_MEGA_cal, users_MEGA_cal = run_simulation_mu_changes(user_MEGA_col2, N, mu_list, time_end, mega_args)

  # plot collision
  collision_over_time_MEGA = compute_collision(num_users_on_channels_MEGA, N)
  collision_over_time_RAND = compute_collision(num_users_on_channels_RAND, N)
  collision_over_time_MEGA_cal = compute_collision(num_users_on_channels_MEGA_cal, N)
  
  plot_dict("collision_over_time", {'MEGA':collision_over_time_MEGA, 'RAND':collision_over_time_RAND, 'MEGA - adapt' : collision_over_time_MEGA_cal})

  # extract reward per user, then print it
  reward_mat_mega = np.array([user.reward for user in users_MEGA], dtype=np.int8).T
  reward_mat_rand = np.array([user.reward for user in users_RAND], dtype=np.int8).T
  reward_mat_mega_cal = np.array([user.reward for user in users_MEGA_cal], dtype=np.int8).T

  regret_MEGA = compute_regret(mu_list, reward_mat_mega)
  regret_RAND = compute_regret(mu_list, reward_mat_rand)
  regret_MEGA_cal = compute_regret(mu_list, reward_mat_mega_cal)

  plot_dict("regret_over_time", {'MEGA':regret_MEGA, 'RAND':regret_RAND, 'MEGA - adapt' : regret_MEGA_cal})


def main2(seed = 0):
  # mu_list = [0.9, 0.8 ,0.7, 0.6, 0.5 ,0.4, 0.3, 0.2 ,0.1]
  mu_list = [0.9, 0.75 ,0.73, 0.71, 0.42 ,0.4, 0.41, 0.2 ,0.1]
  rng = np.random.default_rng(seed)

  K = len(mu_list)
  N = 6
  c = 0.1
  p_0 = 0.6
  alpha = 0.5
  beta = 0.8
  d = 0.05
  delta = 0.05

  time_end = int(1e4)
  simulation_time = 5
  ############ check ##########
  A = np.pad(np.sort(mu_list), (0, 1))
  B = np.pad(np.sort(mu_list), (1, 0))
  min_distance_between_rewards = np.min((A-B)[:-1])
  if min_distance_between_rewards < d:
    warnings.warn("This is a warning message.")
  
  ############ main loop ##########
  collision_over_time_MEGA = np.zeros([time_end])
  collision_over_time_RAND = np.zeros([time_end])
  collision_over_time_MEGA_cal = np.zeros([time_end])
  regret_MEGA = np.zeros([time_end])
  regret_RAND = np.zeros([time_end])
  regret_MEGA_cal = np.zeros([time_end])
  
  for idx in range(simulation_time):
    print(idx)
    mu_list = rng.permutation(mu_list)
    mega_args = {"p_0": p_0, "alpha": alpha, "beta": beta, "c": c, "d": d, "K": K, "debug_group": True}
    num_users_on_channels_MEGA, users_MEGA = run_simulation(user_MEGA, N, mu_list, time_end, mega_args)
    rand_args = {"K": K}
    num_users_on_channels_RAND, users_RAND = run_simulation(user_RAND, N, mu_list, time_end, rand_args)
    mega_args['delta'] = delta
    num_users_on_channels_MEGA_cal, users_MEGA_groups = run_simulation(user_MEGA_groups, N, mu_list, time_end, mega_args)

    # plot collision
    collision_over_time_MEGA += compute_collision(num_users_on_channels_MEGA, N)
    collision_over_time_RAND += compute_collision(num_users_on_channels_RAND, N)
    collision_over_time_MEGA_cal += compute_collision(num_users_on_channels_MEGA_cal, N)
    
    # time simulation
    channel_prodiction_error = {'MEGA':np.zeros(time_end), 'MEGA_groups':np.zeros(time_end)}
    channel_order = np.array([user.channel_order for user in users_MEGA])
    channel_prodiction_error['MEGA'] += channel_prodication_error(channel_order, time_end, K)
    channel_order = np.array([user.channel_order for user in users_MEGA_groups])
    channel_prodiction_error['MEGA_groups'] += channel_prodication_error(channel_order, time_end, K)

    # extract reward per user, then print it
    reward_mat_mega = np.array([user.reward for user in users_MEGA], dtype=np.int8).T
    reward_mat_rand = np.array([user.reward for user in users_RAND], dtype=np.int8).T
    reward_mat_mega_cal = np.array([user.reward for user in users_MEGA_groups], dtype=np.int8).T

    regret_MEGA += compute_regret(mu_list, reward_mat_mega)
    regret_RAND += compute_regret(mu_list, reward_mat_rand)
    regret_MEGA_cal += compute_regret(mu_list, reward_mat_mega_cal)

  toprint = False
  plot_dict("collision_over_time", {'MEGA':collision_over_time_MEGA/simulation_time, 'RAND':collision_over_time_RAND/simulation_time, 'MEGA_group' : collision_over_time_MEGA_cal/simulation_time}, save=toprint)
  plot_dict("regret_over_time", {'MEGA':regret_MEGA/simulation_time, 'RAND':regret_RAND/simulation_time, 'MEGA_group' : regret_MEGA_cal/simulation_time}, save=toprint)
  plot_dict("channel_prodiction_error", channel_prodiction_error, save=toprint)
  
  # save
  # save_matrix_with_mu(num_users_on_channels_MEGA, mu_list, f'MEGA_{time_end}_num_users_on_channel')
  # save_matrix_with_mu(num_users_on_channels_MEGA_cal, mu_list, f'MEGA_group_{time_end}_num_users_on_channel')
  # save_matrix_with_mu(num_users_on_channels_RAND, mu_list, f'RAND_{time_end}_num_users_on_channel')

  # user_idx_list = list(range(N))
  # save_matrix_with_mu(reward_mat_mega, user_idx_list, f'MEGA_{time_end}_reward_per_user')
  # save_matrix_with_mu(reward_mat_mega_cal, user_idx_list, f'MEGA_group_{time_end}_reward_per_user')
  # save_matrix_with_mu(reward_mat_rand, user_idx_list, f'RAND_{time_end}_reward_per_user')
  
  # save_matrix_with_mu(channel_prodiction_error['MEGA'], user_idx_list, f'MEGA_{time_end}_channel_prodiction_error')
  # save_matrix_with_mu(channel_prodiction_error['MEGA_group'], user_idx_list, f'MEGA_group_{time_end}_channel_prodiction_error')
  # save_matrix_with_mu(channel_prodiction_error['RAND'], user_idx_list, f'RAND_{time_end}_channel_prodiction_error')
  

if __name__ == "__main__":
  seed = 74406 # normal cal
  seed = 8905
  seed = 9800 # adapt
  seed = 5
  random.seed(seed)
  np.random.seed(seed)
  main3(seed)