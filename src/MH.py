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
    if jokerIndex: 
        J[jokerIndex] = 2

    return C, M, E, J, config['Penalty']

def get_sol_value(C, M, P, J, nHost): 
    """
    Calculate the earning value for a given fight combinaison C and 
    instance parameters M, nHost, and P.
    """
    fightScore = np.sum(C * (M * J))
    penaltyTot = P * (nHost - np.sum(np.max(C, axis=0)))
    return fightScore - penaltyTot

def get_energy_value(C, E):
    """
    Check if energy budget is respected.
    """
    return np.sum(C @ E)



# ----------------------------------------------------------------
#                Neigbors Processing Functions
# ----------------------------------------------------------------

def moove_add_drop(C, E, energy_budget, captain_idx=None):
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
            
    return new_C


def moove_swap_host(C, E, energy_budget): 
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
    

def moove_shift_contestant(C, captain_idx=None):
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


def get_neigbor(C, energy_budget, E, probs, captain_idx=None): 
    """
    Choose a neigbor by processing one action : 
        - Swap : exchange an host from a contestant with a non fought host
        - Shift : transfers a fought host to another free contestant 
        - Add/Drop : add or drop a fight
    depending on the probability of each operation given by the probs vector. 
    """
    u = npr.rand()
    if u <= probs[0]: 
        # Swap
        new_C = moove_swap_host(C, E, energy_budget)
    elif u <= probs[1]: 
        # Shift 
        new_C = moove_shift_contestant(C, captain_idx)
    else: 
        # Add/Drop
        new_C = moove_add_drop(C, E, energy_budget, captain_idx)
    
    return new_C


# ----------------------------------------------------------------
#                       Local Search
# ----------------------------------------------------------------

def local_search_MH(init_C, M, E, P, J, energy_budget, config, nIter, probs, captain_idx=None): 
    nHost = config['nHost']
    C_values = [get_sol_value(init_C, M, P, J, nHost)]
    C = init_C.copy()

    for _ in range(nIter): 
        new_C = get_neigbor(C, energy_budget, E, probs, captain_idx)
        new_C_value = get_sol_value(new_C, M, P, J, nHost)
        if new_C_value > get_sol_value(C, M, P, J, nHost): 
            C = new_C 
        C_values.append(new_C_value)
    
    return C, C_values

# ----------------------------------------------------------------
#                       Execution
# ----------------------------------------------------------------


config = extract_instance(display = False)
C, M, E, J, P = init_matrix(config, None, None)

nIter = 500
probs = [0.3, 0.3, 0.4]
C, C_values = local_search_MH(
    C, M, E, P, J, config['Budget'], config, nIter, probs, None
)

print("Solution Finale : ", C)
plt.plot([i for i in range(0, nIter + 1)], C_values)
plt.show()

