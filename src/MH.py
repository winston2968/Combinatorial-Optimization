# ======================================================================================================
#                                         MH Solving 
# ======================================================================================================


import time 
import sys 
import os 
import numpy as np
from numpy import random as npr
from itertools import product
from matplotlib import pyplot as plt

SOURCE_FOLDER = "src/instances/"
ARG = sys.argv[1]

# ----------------------------------------------------------------
#                       Extract Instance 
# ----------------------------------------------------------------

# Testing args 
FILE_PATH = SOURCE_FOLDER + ARG 
if not os.path.isfile(FILE_PATH):
    print(f"/!\\ Error : file {FILE_PATH} doest not exists ! ")
    print("Stopping...")

def extract_instance(path=FILE_PATH, display=False):
    with open(path, 'r') as file:
        lines = [l.strip() for l in file.readlines() if l.strip() and not l.startswith("#")]

    # Reading global parameters
    nContestant = int(lines[0])
    nHost = int(lines[1])
    energyBudget = int(lines[2])
    penalty = int(lines[3])

    # Contestant lines 
    contestantCompValue = []
    idx = 4
    for _ in range(nContestant):
        parts = lines[idx].split()
        contestantCompValue.append(int(parts[1]))
        idx += 1

    # Host lines 
    hosts_data = []
    for i in range(idx, len(lines)):
        hosts_data.extend(lines[i].split())
    
    hosts = {}
    for i in range(0, len(hosts_data), 5):
        h_id = int(hosts_data[i]) - 1
        hosts[h_id] = hosts_data[i+1 : i+5]

    if display: 
        print(f"NContestant : {nContestant}")
        print(f"NHosts : {nHost}")
        print(f"Energy Budget : {energyBudget}")
        print(f"Penalty : {penalty}")
        print(f"Contestants Values : {contestantCompValue}")
        print(f"Hosts Values : {hosts}")
    return {
        "nContestant" : nContestant, 
        "nHost" : nHost, 
        "Budget" : energyBudget, 
        "Penalty" : penalty, 
        "ContestantValues" : contestantCompValue, 
        "HostValues" : hosts
    }

# ----------------------------------------------------------------
#                       Glocal Functions
# ----------------------------------------------------------------

def init_matrix(config, captainIndex, jokerIndex):
    """
    Create matrix to vectorize operations : 
        - C (Fight config matrix) : (i,j) = 1 iff contestant i fight against host j 
        - W (Win Mask) : (i,j) = W_j if contestant i beats host j 
        - L (Lost Mask) : (i,j) = L_j if host j beats contestant i 
        - M (Gross Earnings Matrix) : (i,j) = W_j if contestant i beats host j 
        else (i,j) = - L_j. And (i,j) = 0 in case of draw. 
        - E (Energy Cost) : vector which represent the energy cost for each host
    We use numpy for all calculation. 
    """
    nHost = config['nHost']
    nCont = config['nContestant']
    # Init fight config matrix C 
    C = np.zeros((nCont, nHost))
    # Init masks
    W = np.zeros((nCont, nHost))
    L = np.zeros((nCont, nHost))
    for i in range(nCont): 
        for j in range(nHost): 
            hValue = int(config['HostValues'][j][0])
            cValue = int(config['ContestantValues'][i])
            if captainIndex == i: 
                cValue += 5 
            if hValue > cValue:
                # Increment loss value 
                L[i,j] = int(config['HostValues'][j][2])
            if hValue < cValue: 
                # Increment win value
                W[i,j] = int(config['HostValues'][j][1])
    # Gross earning matrix 
    M = W - L 
    # Energy cost
    E = np.zeros((nHost, 1))
    for j in range(nHost): 
        E[j,0] = int(config['HostValues'][j][-1])
    # Init Joker vector 
    J = np.ones((nCont, 1))
    if jokerIndex is not None: 
        J[jokerIndex] = 2

    return C, M, E, J, config['Penalty']

def get_sol_value(C, M, P, J, nHost): 
    """
    Calculate the earning value for a given fight combinaison C and 
    instance parameters M, nHost, and P.
    """
    if np.any(np.sum(C, axis=0) > 1):
        return -999999  # Very low score to reject this neighbor
        
    fightScore = np.sum(C * (M * J))
    hosts_fought = np.sum(np.max(C, axis=0))
    penaltyTot = P * (nHost - hosts_fought)
    return fightScore - penaltyTot

def get_energy_value(C, E):
    """
    Check if energy budget is respected.
    """
    return np.sum(np.max(C, axis=0) * E.flatten())



# ----------------------------------------------------------------
#                Neigbors Processing Functions
# ----------------------------------------------------------------

def move_add_drop(C, E, energy_budget, captain_idx=None):
    """
    Add/Drop a fight depending on the value of used energy. 
    """

    new_C = C.copy()
    current_energy = get_energy_value(new_C, E)

    # DROP Logic: Energy is high, we remove a fight
    if current_energy >= 0.95 * energy_budget: 
        active_fights = np.argwhere(new_C == 1)
        if len(active_fights) > 0:
            # Pick a random index among existing fights
            idx = np.random.randint(len(active_fights))
            row, col = active_fights[idx]
            new_C[row, col] = 0 
            
    # ADD Logic: Energy available, we add a fight
    else: 
        # Get free hosts (columns with all zeros)
        free_hosts = np.where(np.max(new_C, axis=0) == 0)[0]
        
        # Determine capacity per contestant (Captain: 1, Others: 2)
        fights_per_cont = np.sum(new_C, axis=1)
        # Create a limit mask
        limits = np.full(len(fights_per_cont), 2)
        if captain_idx is not None:
            limits[captain_idx] = 1 # Captain constraint 
            
        available_conts = np.where(fights_per_cont < limits)[0]

        if len(free_hosts) > 0 and len(available_conts) > 0:
            host = np.random.choice(free_hosts)
            cont = np.random.choice(available_conts)
            new_C[cont, host] = 1
            if get_energy_value(new_C, E) > energy_budget:
                return C # Reject solution if budbet is exceeded
            
    return new_C


def move_swap_host(C, E, energy_budget): 
    """
    Exchange a fought host with a completely free host for a given contestant.
    """
    # Get coordinates of all active fights (row i, col j)
    active_fights = np.argwhere(C == 1)
    if len(active_fights) == 0:
        return C
    # Shuffle the list of fights to try different ones
    np.random.shuffle(active_fights)

    # Get all hosts that are NOT fought by anyone (empty columns)
    free_hosts = np.where(np.max(C, axis=0) == 0)[0]
    if len(free_hosts) == 0:
        return C

    # Try swaps until one respects the energy budget
    for cont_idx, current_host_idx in active_fights:
        # Pick a random free host
        new_host_idx = np.random.choice(free_hosts)
        
        # Calculate energy change: - old host cost + new host cost
        # This is faster than re-calculating the whole energy of the matrix
        energy_diff = E[new_host_idx] - E[current_host_idx]
        current_energy = np.sum(np.max(C, axis=0) * E)
        
        if current_energy + energy_diff <= energy_budget:
            new_C = C.copy()
            new_C[cont_idx, current_host_idx] = 0
            new_C[cont_idx, new_host_idx] = 1
            return new_C

    return C # Return original if no valid swap found
    

def move_shift_contestant(C, captain_idx=None):
    """
    Transfer an existing host from contestant i to contestant k.
    This move is energy-neutral as the host remains the same.
    """
    new_C = C.copy()
    
    # Get all currently active fights (row i, col j)
    active_fights = np.argwhere(C == 1)
    if len(active_fights) == 0:
        return C
    # Shuffle to try different fights
    np.random.shuffle(active_fights)

    # Determine the combat capacity for each contestant
    fights_per_cont = np.sum(C, axis=1)
    # Default limit is 2, but captain is limited to 1
    limits = np.full(len(fights_per_cont), 2)
    if captain_idx is not None:
        limits[captain_idx] = 1

    # Try to find a valid shift
    for old_cont_idx, host_idx in active_fights:
        # Find contestants who still have capacity
        available_conts = np.where(fights_per_cont < limits)[0]
        
        # Remove the current fighter from candidates for this specific host
        available_conts = available_conts[available_conts != old_cont_idx]
        
        if len(available_conts) > 0:
            # Pick a random valid new contestant
            new_cont_idx = np.random.choice(available_conts)
            
            # Perform the shift
            new_C[old_cont_idx, host_idx] = 0
            new_C[new_cont_idx, host_idx] = 1
            return new_C

    return C # Return original if no valid shift is possible

def move_joker(J): 
    """
    Moove the Joker to another position. 
    """
    new_J = np.ones_like(J)
    n_cont = len(J)
    new_joker_idx = np.random.randint(0, n_cont)
    new_J[new_joker_idx] = 2
    return new_J, new_joker_idx

def move_captain(C, current_captain_idx):
    """
    Moove captain and verify that his nb of fight is respected. 
    """
    n_cont = C.shape[0]
    # Choose a new different captain
    new_captain_idx = np.random.choice([i for i in range(n_cont) if i != current_captain_idx])
    
    new_C = C.copy()
    # Check if the new captain has more fight 
    fights = np.where(new_C[new_captain_idx] == 1)[0]
    if len(fights) > 1:
        # Delete a random fight to respect the limit of 1
        drop_idx = np.random.choice(fights)
        new_C[new_captain_idx, drop_idx] = 0
        
    return new_C, new_captain_idx


def get_neigbor(C, J, energy_budget, E, probs, captain_idx=None):
    """
    Choose a neigbor by processing one action :
        - Swap : exchange an host from a contestant with a non fought host
        - Shift : transfers a fought host to another free contestant
        - Add/Drop : add or drop a fight
    depending on the probability of each operation given by the probs vector.
    """
    u = npr.rand()
    if u <= probs[0]: 
        return move_swap_host(C, E, energy_budget), J, captain_idx
    elif u <= probs[1]: 
        return move_shift_contestant(C, captain_idx), J, captain_idx
    elif u <= probs[2]:
        return move_add_drop(C, E, energy_budget, captain_idx), J, captain_idx
    elif u <= probs[3]: 
        new_J, _ = move_joker(J)
        return C, new_J, captain_idx
    else: 
        new_C, new_cap = move_captain(C, captain_idx)
        return new_C, J, new_cap

# ----------------------------------------------------------------
#                       Simple Local Search
# ----------------------------------------------------------------

def local_search_MH(config, nIter, probs, captain_idx=None, joker_idx=None, verbose=True): 
    """
    Perform a local search depending on the config and given 
    Joker and Captain index. 
    """
    nHost = config['nHost']
    energy_budget = config['Budget']

    # Init instance variables
    C, M, E, J, P = init_matrix(config, captain_idx, joker_idx)
    
    # Store the current best score to avoid recomputing it
    current_value = get_sol_value(C, M, P, J, nHost)
    C_values = [current_value]

    if verbose: print("Launching simple local search...")
    
    t1 = time.time()
    for _ in range(nIter): 
        # Get a neighbor (could be a change in C or a change in J)
        new_C, new_J, new_cap = get_neigbor(C, J, energy_budget, E, probs, captain_idx)
        
        # Calculate the score of this new state (New C AND New J)
        new_value = get_sol_value(new_C, M, P, new_J, nHost)
        
        # Standard Hill-Climbing: accept if better or equal
        if new_value >= current_value:
            C = new_C
            J = new_J
            captain_idx = new_cap
            current_value = new_value
            
        C_values.append(current_value) # Track the best score so far
    t2 = time.time()
    
    if verbose: 
        print(f"Best config found with a value : {np.max(C_values)}")
        plt.plot([i for i in range(0, nIter + 1)], C_values)
        plt.title("Solutions Values Evolution")
        plt.xlabel("Iterations")
        # plt.show()
    return C, J, C_values, t2 - t1

# ----------------------------------------------------------------
#                       Captain Local Search
# ----------------------------------------------------------------

def captain_local_search(config, nIter, probs, joker_idx=None, verbose=True): 
    C_values_trajs = []
    C_traj = []
    nCont = config['nContestant']

    if verbose: print("Launching Captain Local Search...")
    t1 = time.time()
    # Choose a different captain for each starting 
    for cap in range(nCont): 
        if verbose: print(f"Processing captain {cap}/{nCont}")
        sol, _, sol_values, _= local_search_MH(config, nIter, probs, cap, joker_idx, False)
        C_traj.append(sol)
        C_values_trajs.append(sol_values)
    t2 = time.time()
    if verbose:
        print(f"Best config found with a value : {np.max(C_values_trajs)}")
        print(f"Executing time : {t2 - t1}s")
        for cap in range(nCont): 
            plt.plot([i for i in range(0, nIter + 1)], C_values_trajs[cap], label=f"Captain {cap}")
        if nCont <= 15: plt.legend()
        plt.title("Sol Values depending on captain")
        plt.xlabel("Iterations")
        plt.ylabel("Solutions Values")
        plt.show()
    
    return C_values_trajs, t2 - t1


# ----------------------------------------------------------------
#                    Simulated Annealing
# ----------------------------------------------------------------

def simulated_annealing(C, config, nIter, probs, T_start=100, cooling_rate=0.99, captain_idx=None, joker_idx=None):
    """
    Implement the simulated annealing to pass throught 
    local optimum.
    """
    nHost = config['nHost']
    energy_budget = config['Budget']
    traj = []

    # Initialization
    _, M, E, J, P = init_matrix(config, captain_idx, joker_idx)
    current_value = get_sol_value(C, M, P, J, nHost)
    best_C, best_J = C.copy(), J.copy()
    best_value = current_value
    
    T = T_start
    C_values = [current_value]

    t1 = time.time()

    print("=> Launching Simulated Annealing...")
    for _ in range(nIter):
        # Get a neighbor
        new_C, new_J, new_cap = get_neigbor(C, J, energy_budget, E, probs, captain_idx)
        new_value = get_sol_value(new_C, M, P, new_J, nHost)
        traj.append(new_value)
        
        # Acceptance logic
        delta = new_value - current_value
        
        # If better, we accept. If worse, we accept with a probability
        if delta > 0:
            C, J, captain_idx = new_C, new_J, new_cap
            current_value = new_value
        else:
            # Simulated Annealing process
            acceptance_prob = np.exp(delta / T)
            if np.random.rand() < acceptance_prob:
                C, J = new_C, new_J
                current_value = new_value
        
        # Keep track of the absolute best solution found so far
        if current_value > best_value:
            best_value = current_value
            best_C, best_J = C.copy(), J.copy()
            
        C_values.append(current_value)
        
        # Cooling schedule
        T *= cooling_rate
    t2 = time.time()

    print(f"Best value found : {best_value}")
    print(f"Executing time : {t2 - t1}s")

    return best_C, best_J, C_values, t2 - t1, traj, captain_idx


def greedy_sol_init(config, captain_idx, joker_idx): 
    """
    Init C matrix with fight which max the sol 
    value. 
    """

    # Init matrix 
    C, M, E, J, P = init_matrix(config, captain_idx, joker_idx)
    energy_budget = config['Budget']
    nCont, nHost = M.shape
    
    # Process ratio Gain / Energy for each fight
    gains = M * J
    costs = E.flatten()
    ratios = gains / (costs + 1e-9)

    # Create an array for all possible fight and sort it 
    # by ratio
    all_fights = []
    for i in range(nCont):
        for j in range(nHost):
            if ratios[i, j] > 0:
                all_fights.append((i, j, ratios[i, j]))

    all_fights.sort(key=lambda x: x[2], reverse=True)

    # Greedy fill-in 
    current_energy = 0
    for cont, host, ratio in all_fights:
        host_cost = costs[host]
        
        # Check budget
        if current_energy + host_cost <= energy_budget:
            if np.sum(C[:, host]) == 0:
                limit = 1 if cont == captain_idx else 2
                if np.sum(C[cont, :]) < limit:
                    C[cont, host] = 1
                    current_energy += host_cost
                    
    return C
                        


# ----------------------------------------------------------------
#                       Execution
# ----------------------------------------------------------------


config = extract_instance(display = False)

nIter = 2000
probs = [0.35, 0.6, 0.9, 0.95]
nCont = config['nContestant']
nHost = config['nHost']
T_start = 100
cooling_rate = 0.99
joker_idx = 0
captain_idx = 0 

sols = {}
deltaT = 0
C = greedy_sol_init(config, captain_idx, joker_idx)
best_C, best_J, C_values, sub_time, traj, captain_idx = simulated_annealing(C, config, nIter, probs, T_start, cooling_rate, captain_idx, joker_idx)

# plt.plot([i for i in range(len(C_values))], C_values)
# plt.title("Values des configurations retenues en fonction des itérations")
# plt.xlabel("Itérations")
# plt.ylabel("Valeurs de C retenues")
# plt.show()
