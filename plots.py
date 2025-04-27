import numpy as np
import matplotlib.pyplot as plt

## plot regret ovetime
def compute_regret(mu_list:list, reward_mat:np.ndarray) ->np.ndarray:
  time_end, N = reward_mat.shape
  if len(mu_list.shape) == 1:
    mu_list = mu_list.reshape(1,-1)
  
  # get sum best mu over time
  mu_best_list = -np.sort(-mu_list, axis=1)[:,:N]
  sum_mu_best = np.sum(mu_best_list, axis=1)

  # avg reward per timestep
  all_reward = np.sum(reward_mat, axis=1)

  # average over time
  avg_reawrd_over_time_MEGA = np.cumsum(all_reward) / np.arange(1, time_end + 1)

  # regret
  return (sum_mu_best - avg_reawrd_over_time_MEGA)/N

def compute_collision(num_users_on_channels, N):
  # plot collision
  all_collision = np.sum(np.maximum(0, num_users_on_channels - 1), axis=1)/N
  collision_over_time = np.cumsum(all_collision)
  return collision_over_time

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from datetime import datetime

def plot_dict(title: str, vec_dict: dict, xlabel: str = "Time", ylabel: str = "Value", grid_on: bool = True, save: bool = False) -> None:
    """
    Plot a dictionary of vectors with labels and title.
    
    Args:
    - vec_dict: A dictionary where each value is a list or NumPy array to be plotted.
    - title: Title of the plot.
    - xlabel: Label for the x-axis (default: "Time").
    - ylabel: Label for the y-axis (default: "Value").
    - grid_on: Whether to show the grid (default: True).
    - save: Whether to save the plot as an image (default: False).
    """
    for key in vec_dict:
        plt.plot(vec_dict[key], label=key)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(grid_on)

    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"output/{title.replace(' ', '_')}_{timestamp}.png"
        plt.savefig(filename)
        print(f"Plot saved as '{filename}'")
        
    plt.show()
    
    
import matplotlib.pyplot as plt

def plot_subplots(main_title, columns_dict, shape):
  rows, cols = shape
  total_plots = rows * cols

  if len(columns_dict) != total_plots:
    raise ValueError(f"Expected {total_plots} arrays, but got {len(columns_dict)}.")

  fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
  fig.suptitle(main_title)

  axs = axs.flatten()  # Flatten in case of multi-row/col layout

  for i, (idx, array) in enumerate(sorted(columns_dict.items())):
    axs[i].plot(array)
    axs[i].set_title(f"Plot {idx}")
    axs[i].grid(True)

  plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit main title
  plt.show()