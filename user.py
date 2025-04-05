import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings

class user_MEGA:
    def __init__(self, p_0:float, alpha:float, beta:float, c:float, d:float, K:int, time_end:int) -> None:
        self.p_0 = p_0
        self.p = p_0
        self.alpha = alpha
        self.beta = beta
        self.c = c
        self.d = d

        self.K = K
        self.t = 0
        self.taken = np.ones([K],dtype=int)

        # max K is max_int8
        max_int8 = np.iinfo(np.int8).max
        if K > max_int8:
            raise ValueError
        
        self.reward_empirical_mean = {'reward_sum':np.zeros([K]), 'num_sum':np.zeros([K],dtype=np.int32)}
        self.a = np.zeros([time_end], dtype=np.int8)
        self.a[0] = np.random.randint(0, K)

        # udpated code, count for each channel the reward seperatly 
        self.exploit_loc = np.zeros([time_end], dtype=np.int8) - 1

    def step(self, arms_reward_old, num_users_on_channels):
        # get const
        alpha = self.alpha
        beta = self.beta
        K = self.K
        d = self.d
        c = self.c
        # update t
        self.t += 1
        t = self.t

        ################ check reward for t-1 ################
        # if user transmit at t-1
        if self.a[t-1] >= 0:
            # if there is collision
            if num_users_on_channels[t-1, self.a[t-1]] > 1:
                # epsilon-greedy part
                if np.random.rand() < self.p:
                    # persist
                    self.a[t] = self.a[t-1]
                    return
                else:
                    # give up
                    self.taken[self.a[t-1]] = np.random.randint(t, int(t + t**beta))
                    self.p = self.p_0
            
            # else, get reward
            else:
                self.p = self.p*alpha + (1-alpha)
                self.reward_empirical_mean['reward_sum'][self.a[t-1]] += arms_reward_old[self.a[t-1]]
                self.reward_empirical_mean['num_sum'][self.a[t-1]] += 1

        ################ indentify available arms ################
        free_arms = np.where(self.taken <= t)[0]
        if not len(free_arms):
            self.a[t] = -1
            return # Refrain from transmitting in this round
        
        ################ explore or exploit ################
        epsilon = np.min([1, (c*K**2)/(d**2 * (K-1) *t)])
        if np.random.rand() <= epsilon:
            # explore
            self.a[t] = free_arms[np.random.randint(0, len(free_arms))]
        else:
            # exploit
            reward_mean = self.reward_empirical_mean['reward_sum'] / self.reward_empirical_mean['num_sum']
            self.a[t] = np.argsort(reward_mean)[::-1][free_arms][0]
        # update p if a[t] change
        if self.a[t] != self.a[t-1]:
            self.p = self.p_0

class user_RAND:
    def __init__(self, K:int, time_end:int) -> None:
        self.K = K
        self.t = 0
        self.taken = np.ones([K],dtype=int)

        # max K is max_int8
        max_int8 = np.iinfo(np.int8).max
        if K > max_int8:
            raise ValueError
        
        self.a = np.zeros([time_end], dtype=np.int8)
        self.a[0] = np.random.randint(0, K)

        # udpated code, count for each channel the reward seperatly 
        self.exploit_loc = np.zeros([time_end], dtype=np.int8) - 1

    def step(self, arms_reward_old, num_users_on_channels):
        # update t
        self.t += 1
        t = self.t

        ################ check reward for t-1 ################
        # if there is collision
        if num_users_on_channels[t-1, self.a[t-1]] > 1:
            self.a[t] = np.random.randint(0, self.K)
        else:
            self.exploit_loc[t-1] = self.a[t-1] if arms_reward_old[self.a[t-1]] else -2
            self.a[t] = self.a[t-1]

def run_simulation_MEGA(p_0, alpha, beta, c, d, K, time_end, N, mu_list):
    users = [user_MEGA(p_0, alpha, beta, c, d, K, time_end) for idx in range(N)]
    num_users_on_channels = np.zeros([time_end, K], dtype=np.int8)
    for t in tqdm(range(1, time_end)):
        # check collision
        for user in users:
            if user.a[t-1] >=0: num_users_on_channels[t-1, user.a[t-1]] += 1
        # Generate Bernoulli random reward
        arms_reward_old = np.random.binomial(n=1, p=mu_list, size=len(mu_list))
        for user in users:
            user.step(arms_reward_old, num_users_on_channels)
            
    return num_users_on_channels, users

def run_simulation_RAND(K, time_end, N, mu_list):
    users = [user_RAND(K, time_end) for idx in range(N)]
    num_users_on_channels = np.zeros([time_end, K], dtype=np.int8)
    for t in tqdm(range(1, time_end)):
        # check collision
        for user in users:
            if user.a[t-1] >=0: num_users_on_channels[t-1, user.a[t-1]] += 1
        # Generate Bernoulli random reward
        arms_reward_old = np.random.binomial(n=1, p=mu_list, size=len(mu_list))
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

    num_users_on_channels_MEGA, users_MEGA = run_simulation_MEGA(p_0, alpha, beta, c, d, K, time_end, N, mu_list)
    mu_list = [0.9, 0.8 ,0.7, 0.62, 0.5 ,0.4, 0.3, 0.2 ,0.1]
    num_users_on_channels_RAND, users_RAND = run_simulation_MEGA(p_0, alpha, beta, c, d, K, time_end, N, mu_list)

    # plot
    all_collision_MEGA = np.sum(np.maximum(0, num_users_on_channels_MEGA - 1), axis=1)/N
    collision_over_time_MEGA = np.cumsum(all_collision_MEGA)
    
    all_collision_RAND = np.sum(np.maximum(0, num_users_on_channels_RAND - 1), axis=1)/N
    collision_over_time_RAND = np.cumsum(all_collision_RAND)
    
    plt.plot(collision_over_time_MEGA, label='MEGA - original d < Delta')
    plt.plot(collision_over_time_RAND, label='MEGA - modify   d > Delta')
    plt.title("collision_over_time")
    plt.legend()
    plt.show()
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 8))
    axes = axes.flatten()

    for i, user in enumerate(users_MEGA):
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
