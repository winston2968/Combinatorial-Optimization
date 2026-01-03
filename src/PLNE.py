# ======================================================================================================
#                                         PLNE Solving 
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


def solve_PLNE_without_captain(model, config, fight_config, fight_host): 
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

def solve_PLNE_full(model, config, fight_config, fight_host): 
    """
    Solves the PLNE problem with Captain bonus, Joker multiplier, 
    Energy budget, and Host penalties.
    """
    nContestant = config['nContestant']
    nHosts = config['nHosts']
    P = config['Penalty']
    S = 5  # Captain competence bonus

    # Retrieve energy costs from host data [cite: 1]
    fight_cost = [int(config['Hosts_Values'][j][3]) for j in range(nHosts)]

    # 1. Pre-calculate four gain matrices to keep the objective linear
    g_norm, g_cap, g_jok, g_both = [], [], [], []
    
    for i in range(nContestant):
        r_n, r_c, r_j, r_b = [], [], [], []
        c_comp = int(config['Contestants_Values'][i])
        for j in range(nHosts):
            h_comp = int(config['Hosts_Values'][j][0])
            w = int(config['Hosts_Values'][j][1])
            l = int(config['Hosts_Values'][j][2])
            
            # Helper to calculate gain based on competence and multiplier [cite: 2]
            def calc_gain(comp, multiplier):
                if comp > h_comp:
                    return w * multiplier
                elif comp < h_comp:
                    return -l * multiplier
                return 0

            r_n.append(calc_gain(c_comp, 1))      # Normal fight
            r_c.append(calc_gain(c_comp + S, 1))  # Captain bonus (+5)
            r_j.append(calc_gain(c_comp, 2))      # Joker bonus (x2)
            r_b.append(calc_gain(c_comp + S, 2))  # Both bonuses active
            
        g_norm.append(r_n)
        g_cap.append(r_c)
        g_jok.append(r_j)
        g_both.append(r_b)

    # 2. Decision Variables for roles
    isCap = [model.addVar(f"isCap_{i}", vtype="B") for i in range(nContestant)]
    hasJok = [model.addVar(f"hasJok_{i}", vtype="B") for i in range(nContestant)]
    
    # 3. Linearization variables for combined states
    # isCF: i fights j AND i is Captain
    # isJF: i fights j AND i has Joker
    # isBF: i fights j AND i is Captain AND i has Joker
    isCF = [[model.addVar(f"isCF_{i}_{j}", vtype="B") for j in range(nHosts)] for i in range(nContestant)]
    isJF = [[model.addVar(f"isJF_{i}_{j}", vtype="B") for j in range(nHosts)] for i in range(nContestant)]
    isBF = [[model.addVar(f"isBF_{i}_{j}", vtype="B") for j in range(nHosts)] for i in range(nContestant)]

    # 4. Global Role Constraints
    model.addCons(sum(isCap[i] for i in range(nContestant)) <= 1)   # Max 1 captain
    model.addCons(sum(hasJok[i] for i in range(nContestant)) <= 1)  # Max 1 Joker card

    for i in range(nContestant):
        # Rule: Captain fights max 1 time, others max 2 [cite: 2]
        model.addCons(sum(fight_config[i][j] for j in range(nHosts)) <= 2 - isCap[i])
        
        for j in range(nHosts):
            # Linearization for Captain AND Fight
            model.addCons(isCF[i][j] <= fight_config[i][j])
            model.addCons(isCF[i][j] <= isCap[i])
            model.addCons(isCF[i][j] >= fight_config[i][j] + isCap[i] - 1)
            
            # Linearization for Joker AND Fight
            model.addCons(isJF[i][j] <= fight_config[i][j])
            model.addCons(isJF[i][j] <= hasJok[i])
            model.addCons(isJF[i][j] >= fight_config[i][j] + hasJok[i] - 1)

            # Linearization for Both AND Fight (Captain AND Joker AND Fight)
            model.addCons(isBF[i][j] <= isCF[i][j])
            model.addCons(isBF[i][j] <= hasJok[i])
            model.addCons(isBF[i][j] >= isCF[i][j] + hasJok[i] - 1)

    # 5. Host and Energy Constraints
    for j in range(nHosts):
        # Each host can be engaged in at most one fight [cite: 2]
        model.addCons(sum(fight_config[i][j] for i in range(nContestant)) == fight_host[j])

    # Rule: Total energy costs must not exceed budget B [cite: 2]
    model.addCons(sum(fight_cost[j] * fight_host[j] for j in range(nHosts)) <= config['Energy_Budget'])

    # 6. Objective Function Calculation
    # We use inclusion-exclusion logic to ensure only one gain matrix applies per fight
    total_fights_gain = sum(
        g_norm[i][j] * (fight_config[i][j] - isCF[i][j] - isJF[i][j] + isBF[i][j]) +
        g_cap[i][j] * (isCF[i][j] - isBF[i][j]) +
        g_jok[i][j] * (isJF[i][j] - isBF[i][j]) +
        g_both[i][j] * isBF[i][j]
        for i in range(nContestant) for j in range(nHosts)
    )
    
    # Penalty for each host that remains unfought [cite: 2]
    total_penalty = P * (nHosts - sum(fight_host))

    model.setObjective(total_fights_gain - total_penalty, "maximize")

    # Optimization Execution
    # model.hideOutput()
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