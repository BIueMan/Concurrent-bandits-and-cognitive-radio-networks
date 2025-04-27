import numpy as np
from functools import wraps

def add_debug_group(cls):
    # --- Patch __init__ ---
    original_init = cls.__init__

    @wraps(original_init)
    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)

        if self.debug_group:
            time_end = kwargs['time_end']
            K = self.K
            self.channel_order = np.zeros([time_end, K], dtype=np.int8)
            self.channel_order[0] = np.arange(K)

    cls.__init__ = new_init

    # --- Patch step() ---
    if hasattr(cls, 'step'):
        original_step = getattr(cls, 'step')

        @wraps(original_step)
        def new_step(self, *args, **kwargs):
            result = original_step(self, *args, **kwargs)
            if self.debug_group:
                t = self.t
                delta = self.delta if hasattr(self, 'delta') else None
                
                with np.errstate(invalid='ignore', divide='ignore'):
                    reward_mean = self.reward_empirical_mean['reward_sum'] / self.reward_empirical_mean['num_sum']
                reward_mean = np.nan_to_num(reward_mean, nan=0.0)
                reward_idx_sort = self._sort_rewards_with_threshold(reward_mean, delta) if delta else np.argsort(reward_mean)
                self.channel_order[t] = reward_idx_sort
            return result

        setattr(cls, 'step', new_step)

    return cls