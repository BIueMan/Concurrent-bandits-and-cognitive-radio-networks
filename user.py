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
            reward_idx_sort = np.argsort(reward_mean)[::-1]
            self.a[t] = self._find_first_valid(reward_idx_sort, free_arms)
        # update p if a[t] change
        if self.a[t] != self.a[t-1]:
            self.p = self.p_0
            
    def _find_first_valid(self, values, valid_set):
        # Convert valid_set to a set for O(1) lookups
        valid = set(valid_set)
        valid_elements = filter(lambda x: x in valid, values)
        return next(valid_elements, None)

class user_MEGA_col:
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
            reward_idx_sort = self._sort_rewards_with_threshold(reward_mean, 0.05)
            self.a[t] = self._find_first_valid(reward_idx_sort, free_arms)
        # update p if a[t] change
        if self.a[t] != self.a[t-1]:
            self.p = self.p_0
        
    def _find_first_valid(self, values, valid_set):
        # Convert valid_set to a set for O(1) lookups
        valid = set(valid_set)
        valid_elements = filter(lambda x: x in valid, values)
        return next(valid_elements, None)
            
    def _sort_rewards_with_threshold(self, rewards:np.ndarray, alpha):
        """
        Sort rewards array from large to small with special handling for close values.
        Uses grouping before sorting for potential efficiency gains.
        
        Parameters:
        rewards (numpy.ndarray): The rewards to sort
        alpha (float): Threshold for considering two values as equal
        
        Returns:
        numpy.ndarray: Indices of the sorted rewards
        """
        # Create pairs (value, index)
        paired = [(val, idx) for idx, val in enumerate(rewards)]
        
        ## Group similar values
        groups = []
        unprocessed = set(range(len(paired)))
        while unprocessed:
            # first unprocessed element
            first_idx = next(iter(unprocessed))
            first_val = paired[first_idx][0]
            
            # Find all elements close to it
            current_group = []
            for i in list(unprocessed):
                if abs(paired[i][0] - first_val) < alpha:
                    current_group.append(paired[i])
                    unprocessed.remove(i)
            
            # Sort group by original index
            current_group.sort(key=lambda x: x[1])
            
            # Add the group with its representative value (use the max value in the group)
            representative_value = max(item[0] for item in current_group)
            groups.append((representative_value, current_group))
        
        # Sort groups by their representative values (descending)
        groups.sort(key=lambda x: x[0], reverse=True)
        
        # Flatten the sorted groups
        result = []
        for _, group in groups:
            result.extend(group)
        
        # Extract just the indices
        sorted_indices = [idx for _, idx in result]
        
        return np.array(sorted_indices)


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
            
