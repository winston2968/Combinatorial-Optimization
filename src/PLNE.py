# ======================================================================================================
#                                         PLNE Version 
# ======================================================================================================


from pyscipopt import Model
import time
import sys
import os 

SOURCE_FOLDER = "src/instances/"
ARG = sys.argv[1]

# Testing args 
FILE_PATH = SOURCE_FOLDER + ARG 
if not os.path.isfile(FILE_PATH):
    print(f"/!\\ Error : file {FILE_PATH} doest not exists ! ")
    print("Stopping...")

def extract_instance(path=FILE_PATH, display=False):
    with open(path, 'r') as file:
        # On ne garde que les lignes qui ne sont pas des commentaires et non vides
        lines = [l.strip() for l in file.readlines() if l.strip() and not l.startswith("#")]

    # Lecture des 4 paramètres globaux (les 4 premières lignes utiles)
    nContestant = int(lines[0])
    nHosts = int(lines[1])
    energyBudget = int(lines[2])
    penalty = int(lines[3])

    # Lecture des combattants (C lignes suivantes)
    contestantCompValue = []
    idx = 4
    for _ in range(nContestant):
        parts = lines[idx].split()
        contestantCompValue.append(int(parts[1]))
        idx += 1

    # Lecture des hôtes (H lignes restantes)
    # Note : Utilisation d'une liste plate pour gérer les sauts de ligne dans les fichiers
    hosts_data = []
    for i in range(idx, len(lines)):
        hosts_data.extend(lines[i].split())
    
    hosts = {}
    # On groupe par 5 (ID, Compétence, Win, Loss, Energy)
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

    # Init matrix to define the win value obtained by contestant team
    win_values = [
        [
            int(config["Hosts_Values"][j][2]) for j in range(config["nHosts"])
        ] for _ in range(config["nContestant"])
    ]

    return M, fights_config, fights_host, win_values


def display_PLNE_Instance(model, config, fights_config, fight_host):
    """
        Display PLNE Model results after optimisation
    """
    # Checking sol status
    if model.getStatus() != "optimal":
        print(f"Model is not optimal (Statut : {model.getStatus()})")

    print("\n======= OPTIMISATION RESULTS =======")
    print(f"Total Gain (Objectif) : {model.getObjVal()}")
    
    print("\n=> Foughts Hosts (ID) :")
    hosts_fought_count = 0
    for j in range(config['nHosts']): 
        if model.getVal(fight_host[j]) > 0.5:
            print(f"Hôte {j+1} ", end="| ")
            hosts_fought_count += 1
    
    print(f"\nTotal : {hosts_fought_count} foughts hosts on {config['nHosts']}.")

    print("\n=> Fights Config (Contestant -> Host) :")
    for i in range(config["nContestant"]): 
        for j in range(config["nHosts"]): 
            if model.getVal(fights_config[i][j]) > 0.5:
                print(f" - Contestant {i+1} confront host {j+1}")

    print("\n===========================================")


def solve_PLNE_1(model, config, fight_config, fight_host, win_values): 
    """
        Solve the PLNE problem with the given constraints
    """

    nContestant = config['nContestant']
    nHosts = config['nHosts']
    P = config['Penalty']

    # Calculate gains values
    gain_coeffs = []
    for i in range(nContestant):
        row = []
        c_comp = int(config['Contestants_Values'][i])
        for j in range(nHosts):
            h_comp = int(config['Hosts_Values'][j][0])
            w_j = int(config['Hosts_Values'][j][1])
            l_j = int(config['Hosts_Values'][j][2])
            
            if c_comp > h_comp:
                row.append(w_j)
            elif c_comp < h_comp:
                row.append(-l_j)
            else:
                row.append(0)
        gain_coeffs.append(row)

    # Correct fight_cost calculation based on Hosts_Values
    fight_cost = [
        int(config['Hosts_Values'][j][-1]) for j in range(nHosts)
    ]

    # Each contestant can engage at most 2 fights
    for i in range(nContestant): 
        model.addCons(sum(fight_config[i][j] for j in range(nHosts)) <= 2)
    
    # Each host can engage at most 1 fight 
    for j in range(nHosts):
        model.addCons(sum(fight_config[i][j] for i in range(nContestant)) <= 1)
    
    # Links between fight_config and fight_host 
    for j in range(nHosts): 
        model.addCons(sum(fight_config[i][j] for i in range(nContestant)) <= fight_host[j])

    # Respect global fight cost
    model.addCons(sum(fight_cost[j] * fight_host[j] for j in range(nHosts)) <= config['Energy_Budget'])

    # Fights gains sum 
    total_fights_gain = sum(gain_coeffs[i][j] * fight_config[i][j] 
                            for i in range(nContestant) for j in range(nHosts))
    
    # Penalty sum : P * (nHOsts not foughts)
    # nHosts - sum(fight_host) give the number of not foughts hosts
    total_penalty = P * (nHosts - sum(fight_host))

    model.setObjective(total_fights_gain - total_penalty, "maximize")

    model.hideOutput()

    # Launching optumisation
    model.optimize()


# --------------------------
# Execution
# --------------------------

config = extract_instance(display=True)

# Initialisation
model, fight_config, fight_host, win_values = init_PLNE_Model(config)

# Solving
solve_PLNE_1(model, config, fight_config, fight_host, win_values)

# Display Results
display_PLNE_Instance(model, config, fight_config, fight_host)