import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
from user import *
from load_save import *
from plots import *

def run_simulation(UserClass,  N, mu_list, time_end, user_class_kwargs):
    # Unpack args
    user_class_kwargs['time_end'] = time_end # time_end needed to know in both server and user (to optimize runtime)

    # create users
    users = [UserClass(**user_class_kwargs) for _ in range(N)]
    num_users_on_channels = np.zeros([time_end, len(mu_list)], dtype=np.int8)

    ##### main loop #####
    for t in tqdm(range(1, time_end)):
        # Track collisions
        for user in users:
            if user.a[t-1] >= 0: num_users_on_channels[t-1, user.a[t-1]] += 1
        # Generate Bernoulli random rewards
        arms_reward_old = np.random.binomial(n=1, p=mu_list, size=len(mu_list))
        for user in users:
            user.step(arms_reward_old, num_users_on_channels)

    return num_users_on_channels, users

def run_simulation_div(UserClass,  N, mu_list, mu_div, time_end, user_class_kwargs):
    # Unpack args
    user_class_kwargs['time_end'] = time_end # time_end needed to know in both server and user (to optimize runtime)

    # create users
    users = [UserClass(**user_class_kwargs) for _ in range(N)]
    num_users_on_channels = np.zeros([time_end, len(mu_list)], dtype=np.int8)
    
    def create_mu_matrix(mu_list, N, div):
        mu_list = np.array(mu_list)
        noise = np.random.uniform(low=-div, high=div, size=(len(mu_list), N))
        matrix = mu_list[:, np.newaxis] + noise
        matrix = np.clip(matrix, 0, 1)
        return matrix
    mu_matrix = create_mu_matrix(mu_list, N, mu_div)

    ##### main loop #####
    for t in tqdm(range(1, time_end)):
        # Track collisions
        for user in users:
            if user.a[t-1] >= 0: num_users_on_channels[t-1, user.a[t-1]] += 1
        # Generate Bernoulli random rewards
        for idx, user in enumerate(users):
            selected_mu_list = mu_matrix[:, idx]
            arms_reward_old = np.random.binomial(n=1, p=selected_mu_list, size=len(selected_mu_list))
            user.step(arms_reward_old, num_users_on_channels)

    return num_users_on_channels, users

def run_simulation_mu_changes(UserClass, N, mu_list, time_end, user_class_kwargs):
    # Unpack args
    user_class_kwargs['time_end'] = time_end # time_end needed to know in both server and user (to optimize runtime)

    # create users
    users = [UserClass(**user_class_kwargs) for _ in range(N)]
    num_users_on_channels = np.zeros([time_end, mu_list.shape[1]], dtype=np.int8)

    ##### main loop #####
    for t in tqdm(range(1, time_end)):
        # Track collisions
        for user in users:
            if user.a[t-1] >= 0: num_users_on_channels[t-1, user.a[t-1]] += 1
        # Generate Bernoulli random rewards
        arms_reward_old = np.random.binomial(n=1, p=mu_list[t])
        for user in users:
            user.step(arms_reward_old, num_users_on_channels)

    return num_users_on_channels, users
            
def test_1():
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

    save_matrix_with_mu(num_users_on_channels_MEGA, mu_list, 'MEGA_num_users_on_channel')
    save_matrix_with_mu(num_users_on_channels_RAND, mu_list, 'RAND_num_users_on_channel')
    
    # extract reward per user
    user_idx_list = list(range(N))
    reward_mat_mega = np.array([user.reward for user in users_MEGA], dtype=np.int8).T
    reward_mat_rand = np.array([user.reward for user in users_RAND], dtype=np.int8).T
    save_matrix_with_mu(reward_mat_mega, user_idx_list, 'MEGA_reward_per_user')
    save_matrix_with_mu(reward_mat_rand, user_idx_list, 'RAND_reward_per_user')

    # plot collision
    collision_over_time_MEGA = compute_collision(num_users_on_channels_MEGA, N)
    collision_over_time_RAND = compute_collision(num_users_on_channels_RAND, N)
    
    plot_dict("collision_over_time", {'MEGA - d < Delta':collision_over_time_MEGA, 'RAND':collision_over_time_RAND})
  
    
    regret_MEGA = compute_regret(mu_list, reward_mat_mega)
    regret_RAND = compute_regret(mu_list, reward_mat_rand)
    
    plot_dict("regret_over_time", {'MEGA - d < Delta':regret_MEGA, 'RAND':regret_RAND})
    
    all_reward_RAND = np.sum(reward_mat_rand, axis=1)
    reawrd_over_time_RAND = np.cumsum(all_reward_RAND)
    
    plt.plot(collision_over_time_MEGA, label='MEGA - d < Delta')
    plt.plot(collision_over_time_RAND, label='RAND')
    plt.title("collision_over_time")
    plt.legend()
    plt.show()
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 8))
    axes = axes.flatten()

    for i, user in enumerate(users_RAND):
        axes[i].plot(user.a)
        axes[i].set_title(f'user {i+1}')
        axes[i].set_xlabel('time')
        axes[i].set_ylabel('channel')

    plt.tight_layout()
    plt.show()
    
    t = np.arange(1, time_end)
    epsilon = np.minimum(1, (c*K**2)/(d**2 * (K-1) *t))
    plt.plot(epsilon)
    plt.title("epsilon")
    plt.show()
    
    


if __name__ == "__main__":
    test_1()
