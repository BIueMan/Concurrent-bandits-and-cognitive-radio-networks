import numpy as np

np.random.seed(0)

def E_greedy_collision_avoidance(N:int, p_0:float, beta:float, arms:np.ndarray, time_end:int):
    K = arms.shape[0]
    
    eta = np.zeros([N]) # collision indicator [per_user]
    p = p_0 * np.ones([N]) # persistence to staty at arm [per_user]
    a = np.random.randint(0, K, size = [N]) # select random arm to persist [per arm]
    a_old = a
    taken_arm = np.zeros(N, K) # the time user N saying arm k is taken
    
    for t in range(time_end):
        for n in range(N):
            E_greedy_user(t, p_0, beta, arms, time_end)
        
    
def E_greedy_user(t:int, p_0:float, beta:float, eta:float, p:float, arms:np.ndarray, time_end:int):
    K = arms.shape[0]
    
    eta = np.zeros([N]) # collision indicator [per_user]
    p = p_0 * np.ones([N]) # persistence to staty at arm [per_user]
    a = np.random.randint(0, K, size = [N]) # select random arm to persist [per arm]
    a_old = a
    taken_arm = np.zeros(N, K) # the time user N saying arm k is taken
    
    for t in range(time_end):
        epsilon = np.min([1, (c*K**2)/(d**2 * (K-1) *t)])
        
        for n in range(N):
            if eta[n] == 1:
                if np.random.rand() < p[n]:
                    a[n] = a_old[n] # stay persist
                else:
                    # give up on a(t)
                    taken_arm[n, a[n]] = np.random.rand() * (t**beta)
                    p[n] = p_0
                    
            else:
                p = p*a[n]+(1-a[n])
                # update \mu????
            
            # indentify available arms
            free_arms = np.where(taken_arm[n] == 0)[0]
            if not free_arms:
                continue # Refrain from transmitting in this round
            
            # explore
            if np.random.rand() < epsilon:
                a[n] = np.random.randint(0, K)
            # exploit
            else:
                pass
            
            
            if a[n] != a_old[n]:
                p[n] = p_0
                
            # sample a(t)
                