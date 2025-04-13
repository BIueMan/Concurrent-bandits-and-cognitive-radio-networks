import numpy as np
import warnings
from server import *
from user import *
from load_save import *
from plots import *





def main1():
  # mu_list = [0.9, 0.8 ,0.7, 0.6, 0.5 ,0.4, 0.3, 0.2 ,0.1]
  # mu_list = [0.9, 0.72 ,0.71, 0.73, 0.7 ,0.69, 0.68, 0.2 ,0.1]
  mu_list = [0.9, 0.89 ,0.88, 0.87, 0.86 ,0.85, 0.2, 0.2 ,0.2]
  # mu_list = [0.6, 0.55, 0.52, 0.51, 0.5, 0.49, 0.48, 0.47, 0.46, 0.45]
  rng = np.random.default_rng(42)
  mu_list = rng.permutation(mu_list)

  K = len(mu_list)
  N = 6
  c = 0.1
  p_0 = 0.6
  alpha = 0.5
  beta = 0.8
  d = 0.05
  delta = 0.05

  time_end = int(1e4)
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
  mega_args['delta'] = delta
  num_users_on_channels_MEGA_cal, users_MEGA_cal = run_simulation(user_MEGA_col, N, mu_list, time_end, mega_args)

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
  
  # # save
  # save_matrix_with_mu(num_users_on_channels_MEGA, mu_list, 'MEGA_num_users_on_channel')
  # save_matrix_with_mu(num_users_on_channels_RAND, mu_list, 'RAND_num_users_on_channel')

  # user_idx_list = list(range(N))
  # save_matrix_with_mu(reward_mat_mega, user_idx_list, 'MEGA_reward_per_user')
  # save_matrix_with_mu(reward_mat_rand, user_idx_list, 'RAND_reward_per_user')


if __name__ == "__main__":
  main1()