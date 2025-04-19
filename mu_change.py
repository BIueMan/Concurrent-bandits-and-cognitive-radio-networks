import numpy as np

def generate_mu_change(time_end, params):
    """
    Generate a time-varying mu_list for each arm based on sinusoidal functions.

    Parameters:
        time_end (int): Number of time steps.
        params (list of dict): Each dict contains 'power', 'change', 'w', and 'theta' for an arm.

    Returns:
        mu_list (np.ndarray): Array of shape (time_end, num_arms)
    """
    t = np.arange(time_end).reshape(-1, 1)  # shape (time_end, 1)
    num_arms = len(params)
    mu_list = np.zeros((time_end, num_arms))

    for i, p in enumerate(params):
        power = p.get('power', 0.5)
        change = p.get('change', 0.4)
        w = p.get('w', 200)
        theta = p.get('theta', 0)

        mu_list[:, i] = power + change * np.sin(2 * np.pi * t[:, 0] / w + theta)

    # Clamp values between 0 and 1 to ensure valid probabilities
    mu_list = np.clip(mu_list, 0, 1)

    return mu_list

def generate_mu_constant(time_end, params):
    """
    Generate a time-varying mu_list for each arm based on piecewise constant values.

    Parameters:
        time_end (int): Number of time steps.
        params (list of dict): Each dict must contain 'value' and 'time' for an arm.
            - 'value': list of float values for each constant segment.
            - 'time': list of integers where the value changes. Must be strictly increasing.

    Returns:
        mu_list (np.ndarray): Array of shape (time_end, num_arms)
    """
    t = np.arange(time_end)
    num_arms = len(params)
    mu_list = np.zeros((time_end, num_arms))

    for i, p in enumerate(params):
        values = p.get('value')
        times = p.get('time')

        if not isinstance(values, list) or not isinstance(times, list):
            raise ValueError(f"Arm {i}: 'value' and 'time' must be lists.")
        if sorted(times) != times:
            raise ValueError(f"Arm {i}: 'time' values must be in strictly increasing order.")
        if len(values) != len(times) + 1:
            raise ValueError(f"Arm {i}: 'value' list must be one element longer than 'time' list.")
        if times and time_end <= times[-1]:
            raise ValueError(f"Arm {i}: Last time value {times[-1]} exceeds or equals time_end {time_end}.")

        start_idx = 0
        for j, change_time in enumerate(times):
            change_time = int(change_time)
            mu_list[start_idx:change_time, i] = values[j]
            start_idx = change_time
        mu_list[start_idx:, i] = values[-1]

    mu_list = np.clip(mu_list, 0, 1)
    return mu_list