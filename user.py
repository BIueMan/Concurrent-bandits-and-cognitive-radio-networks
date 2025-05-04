import numpy as np
from group_values import *
import random
from CircularAveragingBuffer import *
from group_up_debug import *


@add_debug_group
class user_MEGA:
    def __init__(self, p_0:float, alpha:float, beta:float, c:float, d:float, K:int, time_end:int, debug_group:bool = False) -> None:
        self.p_0 = p_0
        self.p = p_0
        self.alpha = alpha
        self.beta = beta
        self.c = c
        self.d = d
        self.time_end = time_end

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
        
        self.debug_group = debug_group

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
            with np.errstate(invalid='ignore', divide='ignore'): # ignore devide 0/0
                reward_mean = self.reward_empirical_mean['reward_sum'] / self.reward_empirical_mean['num_sum']
            reward_mean = np.nan_to_num(reward_mean, nan=0.0)
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

class user_MEGA_col2:
    def __init__(self, p_0:float, alpha:float, beta:float, c:float, d:float, K:int, time_end:int, avg_size:int, min_explore_value:float) -> None:
        self.p_0 = p_0
        self.p = p_0
        self.alpha = alpha
        self.beta = beta
        self.c = c
        self.d = d
        self.avg_size = avg_size
        self.min_explore_value = min_explore_value

        self.K = K
        self.t = 0
        self.taken = np.ones([K],dtype=int)

        # max K is max_int8
        max_int8 = np.iinfo(np.int8).max
        if K > max_int8:
            raise ValueError
        
        self.reward_empirical_mean = [CircularAveragingBuffer(self.avg_size) for _ in range(K)]
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
                
                self.reward_empirical_mean[self.a[t-1]].add(arms_reward_old[self.a[t-1]])
                
        ################ indentify available arms ################
        free_arms = np.where(self.taken <= t)[0]
        if not len(free_arms):
            self.a[t] = -1
            return # Refrain from transmitting in this round
        
        ################ explore or exploit ################
        epsilon = max(np.min([1, (c*K**2)/(d**2 * (K-1) *t)]), self.min_explore_value)
        if np.random.rand() <= epsilon:
            # explore
            self.a[t] = free_arms[np.random.randint(0, len(free_arms))]
        else:
            # exploit
            with np.errstate(invalid='ignore', divide='ignore'): # ignore devide 0/0
                reward_mean = np.array([buf.average() for buf in self.reward_empirical_mean])
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

@add_debug_group
class user_MEGA_groups:
    def __init__(self, p_0:float, alpha:float, beta:float, c:float, d:float, K:int, time_end:int, delta:float, debug_group:bool = False) -> None:
        self.p_0 = p_0
        self.p = p_0
        self.alpha = alpha
        self.beta = beta
        self.c = c
        self.d = d

        self.K = K
        self.t = 0
        self.taken = np.ones([K],dtype=int)
        
        self.delta = delta

        # max K is max_int8
        max_int8 = np.iinfo(np.int8).max
        if K > max_int8:
            raise ValueError
        
        self.reward_empirical_mean = {'reward_sum':np.zeros([K]), 'num_sum':np.zeros([K],dtype=np.int32)}
        self.reward = np.zeros([time_end], dtype=np.int8)
        self.a = np.zeros([time_end], dtype=np.int8)
        self.a[0] = np.random.randint(0, K)
        
        self.debug_group = debug_group

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
        delta = self.delta

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
            with np.errstate(invalid='ignore', divide='ignore'): # ignore devide 0/0
                reward_mean = self.reward_empirical_mean['reward_sum'] / self.reward_empirical_mean['num_sum']
            reward_mean = np.nan_to_num(reward_mean, nan=0.0)
            reward_idx_sort = self._sort_rewards_with_threshold(reward_mean, delta)
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
        resoults = group_values_by_delta(list(rewards), alpha)
        # for key in resoults:
        #     random.shuffle(resoults[key])
        unpacked_indices = [idx for group_indices in resoults.values() for idx in group_indices]
        
        return unpacked_indices


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
            
class user_EXPLORATION_FACTOR_ONLY:
    def __init__(self, K:int, time_end:int, c:float, d:float) -> None:
        self.K = K
        self.t = 0
        self.c = c
        self.d = d

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
        is_collision = num_users_on_channels[t-1, self.a[t-1]] > 1
        if not is_collision:
            self.reward[t-1] = arms_reward_old[self.a[t-1]]
        
        ## [t] go to next channel while only check the exploration factor
        K = self.K
        c = self.c
        d = self.d
        ################ explore or exploit ################
        epsilon = np.min([1, (c*K**2)/(d**2 * (K-1) *t)])
        if np.random.rand() <= epsilon or is_collision:
            # explore
            self.a[t] = np.random.randint(0, K)
        else:
            # exploit
            self.a[t] = self.a[t-1]