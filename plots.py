import numpy as np
import matplotlib.pyplot as plt

## plot regret ovetime
def compute_regret(mu_list:list, reward_mat:np.ndarray) ->np.ndarray:
  time_end, N = reward_mat.shape
  
  # get sum best mu over time
  mu_best_list = sorted(mu_list, reverse=True)[:N]
  sum_mu_best = np.sum(mu_best_list)

  # avg reward per timestep
  all_reward = np.sum(reward_mat, axis=1)

  # average over time
  avg_reawrd_over_time_MEGA = np.cumsum(all_reward) / np.arange(1, time_end + 1)

  # regret
  return sum_mu_best - avg_reawrd_over_time_MEGA

def compute_collision(num_users_on_channels, N):
  # plot collision
  all_collision = np.sum(np.maximum(0, num_users_on_channels - 1), axis=1)/N
  collision_over_time = np.cumsum(all_collision)
  return collision_over_time

import matplotlib.pyplot as plt

def plot_dict(title: str, vec_dict: dict, xlabel: str = "Time", ylabel: str = "Value", grid_on: str = True) -> None:
    """
    Plot a dictionary of vectors with labels and title.
    
    Args:
    - vec_dict: A dictionary where each value is a list or NumPy array to be plotted.
    - title: Title of the plot.
    - xlabel: Label for the x-axis (default: "Time").
    - ylabel: Label for the y-axis (default: "Value").
    """
    for key in vec_dict:
        plt.plot(vec_dict[key], label=key)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(grid_on)
    plt.show()