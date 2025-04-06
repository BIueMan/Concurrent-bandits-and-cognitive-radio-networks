import numpy as np

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
        self.reward = np.zeros([time_end], dtype=np.int8)
        self.a = np.zeros([time_end], dtype=np.int8)
        self.a[0] = np.random.randint(0, K)

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
                self.reward[t-1] = arms_reward_old[self.a[t-1]]
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

        # max K is max_int8
        max_int8 = np.iinfo(np.int8).max
        if K > max_int8:
            raise ValueError
        
        self.a = np.zeros([time_end], dtype=np.int8)
        self.reward = np.zeros([time_end], dtype=np.int8)
        self.a[0] = np.random.randint(0, K)

    def step(self, arms_reward_old, num_users_on_channels):
        # update t
        self.t += 1
        t = self.t
        
        # [t-1] check if not collision, then get reward
        if not num_users_on_channels[t-1, self.a[t-1]] > 1:
            self.reward[t-1] = arms_reward_old[self.a[t-1]]
        
        ## [t] go to next channel without checking reward or colition
        self.a[t] = np.random.randint(0, self.K)
            
