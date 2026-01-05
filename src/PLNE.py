# ======================================================================================================
#                                         PLNE Solving 
# ======================================================================================================

# Parameters are fixed at the end of the file. 
# To launch and test the code, you just need to change the instance folder path 
# given below and type 'python3 PLNE.py name-of-tournament-file'.

from pyscipopt import Model
from pyscipopt import quicksum
from pyscipopt import SCIP_PARAMSETTING
import numpy as np
import sys
import os

SOURCE_FOLDER = "src/instances/"
ARG = sys.argv[1]

# ----------------------------------------------------------------
#                   Extracting Instance
# ----------------------------------------------------------------

# Testing args 
FILE_PATH = SOURCE_FOLDER + ARG 
if not os.path.isfile(FILE_PATH):
    print(f"/!\\ Error : file {FILE_PATH} doest not exists ! ")
    print("Stopping...")

def extract_instance(path=FILE_PATH, display=False):
    """
    Extract instance from the file. 
    """

    with open(path, 'r') as file:
        # Keep lines without comment
        lines = [l.strip() for l in file.readlines() if l.strip() and not l.startswith("#")]

    # Reading global parameters
    nContestant = int(lines[0])
    nHosts = int(lines[1])
    energyBudget = int(lines[2])
    penalty = int(lines[3])

    # Reading contestant lines 
    contestantCompValue = []
    idx = 4
    for _ in range(nContestant):
        parts = lines[idx].split()
        contestantCompValue.append(int(parts[1]))
        idx += 1

    # Reading host lines 
    hosts_data = []
    for i in range(idx, len(lines)):
        hosts_data.extend(lines[i].split())
    
    hosts = {}
    # Merging host infos
    for i in range(0, len(hosts_data), 5):
        h_id = int(hosts_data[i]) - 1
        hosts[h_id] = hosts_data[i+1 : i+5]

    if display: 
        print(f"NContestant : {nContestant}")
        print(f"NHosts : {nHosts}")
        print(f"Energy Budget : {energyBudget}")
        print(f"Penalty : {penalty}")
        print(f"Contestants Values : {contestantCompValue}")
        print(f"Hosts Values : {hosts}")
    return {
        "nContestant" : nContestant, 
        "nHosts" : nHosts, 
        "Energy_Budget" : energyBudget, 
        "Penalty" : penalty, 
        "Contestants_Values" : contestantCompValue, 
        "Hosts_Values" : hosts
    }


def init_PLNE_Model(config): 
    """
        Init PLNE Model with a given instance
    """

    M = Model()

    # Create a variable matrix to define who fights against who
    fights_config = [
        [
            M.addVar(f"fight_{i}_{j}", vtype="B") for j in range(config["nHosts"])
        ] for i in range(config["nContestant"])
    ]

    # Create a vector which defines if a host fights
    fights_host = [
        M.addVar(f"fightHost_{j}", vtype="B") for j in range(config["nHosts"])
    ]

    return M, fights_config, fights_host


def display_PLNE_Instance(model, config, fights_config, fight_host, isCaptain, hasJok):
    """
    Display PLNE Model results after optimization, including captain details.
    """

    # Checking solution status
    if model.getStatus() != "optimal":
        print(f"Warning: Model is not optimal (Status: {model.getStatus()})")

    print("\n======= OPTIMIZATION RESULTS =======")
    print(f"Total Gain (Objective): {model.getObjVal()}")
    
    # Identify and display the Captain and Joker
    joker_id = None 
    captain_id = None
    for i in range(config['nContestant']):
        if model.getVal(isCaptain[i]) > 0.5:
            captain_id = i + 1
            break
        if model.getVal(hasJok[i]) >= 1: 
            joker_id = i + 1
            break
    
    if captain_id:
        print(f"Designated Captain: Contestant {captain_id} (+5 competence bonus)")
    else:
        print("Designated Captain: None selected")

    if joker_id:
        print(f"Designated Joker: Contestant {joker_id} (x2 win/lose bonus)")
    else:
        print("Designated Joker: None selected")

    # Display Fought Hosts
    print("\n=> Fought Hosts (IDs):")
    hosts_fought_count = 0
    for j in range(config['nHosts']): 
        if model.getVal(fight_host[j]) > 0.5:
            print(f"Host {j+1}", end=" | ")
            hosts_fought_count += 1
    
    print(f"\nTotal: {hosts_fought_count} hosts fought out of {config['nHosts']}.")

    # Display Detailed Fights
    print("\n=> Fights Configuration (Contestant -> Host):")
    for i in range(config["nContestant"]): 
        for j in range(config["nHosts"]): 
            if model.getVal(fights_config[i][j]) > 0.5:
                # Add a label if the contestant is the captain
                cap_label = " (as Captain)" if (i + 1) == captain_id else ""
                print(f" - Contestant {i+1}{cap_label} confronts Host {j+1}")

    print("\n===========================================")


def solve_PLNE_full(model, config, fight_config, fight_host, big_space=False): 
    nContestant = config['nContestant']
    nHosts = config['nHosts']
    P = config['Penalty']
    S = 5  # Captain bonus

    # Gains processing
    g_norm = np.zeros((nContestant, nHosts))
    d_cap = np.zeros((nContestant, nHosts))
    d_jok = np.zeros((nContestant, nHosts))
    
    for i in range(nContestant):
        c_comp = int(config['Contestants_Values'][i])
        for j in range(nHosts):
            h_comp = int(config['Hosts_Values'][j][0])
            w, l = int(config['Hosts_Values'][j][1]), int(config['Hosts_Values'][j][2])
            
            def calc(comp, mult):
                if comp > h_comp: return w * mult
                if comp < h_comp: return -l * mult
                return 0

            gn = calc(c_comp, 1)
            g_norm[i, j] = gn
            d_cap[i, j] = calc(c_comp + S, 1) - gn
            d_jok[i, j] = calc(c_comp, 2) - gn

    # Decision variables
    isCap = [model.addVar(f"isCap_{i}", vtype="B") for i in range(nContestant)]
    hasJok = [model.addVar(f"hasJok_{i}", vtype="B") for i in range(nContestant)]
    
    # Decision variables for bonus
    # actCap[i,j] == 1 iff i fight j AND i is Captain
    actCap = [[model.addVar(f"actCap_{i}_{j}", vtype="B") for j in range(nHosts)] for i in range(nContestant)]
    actJok = [[model.addVar(f"actJok_{i}_{j}", vtype="B") for j in range(nHosts)] for i in range(nContestant)]

    # Role constraints
    model.addCons(quicksum(isCap) <= 1)
    model.addCons(quicksum(hasJok) <= 1)

    for i in range(nContestant):
        # Fights limits : 1 for captain 2 for the others
        model.addCons(quicksum(fight_config[i][j] for j in range(nHosts)) <= 2 - isCap[i])
        
        for j in range(nHosts):
            # Linearization : actCap[i,j] = fight_config[i,j] AND isCap[i]
            model.addCons(actCap[i][j] <= fight_config[i][j])
            model.addCons(actCap[i][j] <= isCap[i])
            
            # Linearisation : actJok[i,j] = fight_config[i,j] AND hasJok[i]
            model.addCons(actJok[i][j] <= fight_config[i][j])
            model.addCons(actJok[i][j] <= hasJok[i])

    # Hosts and energy constraints
    fight_cost = [int(config['Hosts_Values'][j][3]) for j in range(nHosts)]
    for j in range(nHosts):
        model.addCons(quicksum(fight_config[i][j] for i in range(nContestant)) == fight_host[j])

    model.addCons(quicksum(fight_cost[j] * fight_host[j] for j in range(nHosts)) <= config['Energy_Budget'])

    # Objective function
    obj_gain = quicksum(
        g_norm[i,j] * fight_config[i][j] + 
        d_cap[i,j] * actCap[i][j] + 
        d_jok[i,j] * actJok[i][j]
        for i in range(nContestant) for j in range(nHosts)
    )
    
    obj_penalty = P * (nHosts - quicksum(fight_host[j] for j in range(nHosts)))
    model.setObjective(obj_gain - obj_penalty, "maximize")

    # Branch roles first
    for i in range(nContestant):
        model.chgVarBranchPriority(isCap[i], 100)
        model.chgVarBranchPriority(hasJok[i], 100)

    if big_space: 
        # Change solving paramters for big search space 
        model.setRealParam("limits/gap", 0.05)
        model.setPresolve(SCIP_PARAMSETTING.FAST)
        model.setRealParam("limits/time", 60)

    model.optimize()
    return isCap, hasJok

# --------------------------
# Execution
# --------------------------

config = extract_instance(display=True)

# Initialisation
model, fight_config, fight_host = init_PLNE_Model(config)

# Solving
# solve_PLNE_without_captain(model, config, fight_config, fight_host)

# Solving 
isCap, hasJok = solve_PLNE_full(model, config, fight_config, fight_host)

# Display Results
display_PLNE_Instance(model, config, fight_config, fight_host, isCap, hasJok)