# Importing necessary libraries and setting up the initial data and functions
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd
import os

# file save in the same directory as the script
os.chdir(os.path.dirname(__file__))


def antibiotic_decay(A, t_half):
    ke = np.log(2) / t_half
    return -ke * A

def model(y, t, params, f_A, f_C, e_max):
    S, RA, RB, A, E = y
    A = f_A
    B = f_C
    Bmax, KG, FIT, Gmin_A, Gmin_B, HILL_A, HILL_B,  MIC_S, CS_A, CS_B = params

    q = 10**-3
    l = 10**-3
    jN = 10**-6.5

    KG_fit = KG * FIT
    MIC_RA = 3
    MIC_RC = 32

    dSdt = S*(1-((S+RA+RB)/10**Bmax))*KG*(1 - ((KG - Gmin_A/KG)*(A/MIC_S)**HILL_A/((A/MIC_S)**HILL_A - (Gmin_A/KG))) - 
                                          ((KG - Gmin_B/KG)*(B/MIC_S)**HILL_B/((B/MIC_S)**HILL_B - (Gmin_B/KG))))  - jN*S*E
    dRAdt = RA*(1-((S+RA+RB)/10**Bmax))*KG_fit*(1- ((KG - Gmin_A/(KG_fit))*(A/MIC_RA)**HILL_A/((A/MIC_RA)**HILL_A - (Gmin_A/(KG_fit)))) -
                                                 ((KG - Gmin_B/(KG_fit))*(B/(MIC_S*CS_B))**HILL_B/((B/(MIC_S*CS_B))**HILL_B - (Gmin_B/(KG_fit))))) - jN*RA*E
    dRBdt = RB*(1-((S+RA+RB)/10**Bmax))*KG*(1- ((KG - Gmin_A/(KG))*(A/(MIC_S*CS_A))**HILL_A/((A/(MIC_S*CS_A))**HILL_A - (Gmin_A/KG))) -
                                             ((KG - Gmin_B/KG)*(B/MIC_RC)**HILL_B/((B/MIC_RC)**HILL_B - (Gmin_B/KG)))) - jN*RB*E
    
    dAdt = f_A
    dEdt = q*(e_max - E) - l*E

    return [dSdt, dRAdt, dRBdt, dAdt, dEdt]


Bmax = 8
KG = 0.21
FIT = 0.79
Gmin_A = -2
Gmin_B = -0.5
HILL_A = 4
HILL_B = 1
MIC_S = 1
S0 = 10**7
RA0 = 0
RB0 = 0
A0 = 1.2
B0 = 0

length_experiment = 24 * 8
t = np.linspace(0, length_experiment, length_experiment)

# Proceed to creating the combined plot

fig, ax = plt.subplots(figsize=(16, 12))


y0 = [S0, RA0,RB0, A0, 10**2]
solution = [y0]
A_values = [A0]
B_values = [B0]
dt = t[1] - t[0]

CS_A = 0.5
CS_B = 0.5

params = [Bmax, KG, FIT, Gmin_A, Gmin_B, HILL_A, HILL_B,  MIC_S, CS_A, CS_B]

A = 0
B = B0
t_half_AMB = 24
t_half_CAS = 7

e_max = 10**5

list_of_solutions=[]

for s in range(1, 101):
    solution = [y0]
    A_values = [A0]
    B_values = [B0]
    y0 = [S0, RA0, RB0, 1.5, 10**2]
    A = 1.5
    B = 0
    for k in range(1, length_experiment):


        ka = [24, 24*4, 24*5]
        if k in ka:
            A += 1.15  # Application of antibiotic A
        else:
            dA = antibiotic_decay(A, t_half_AMB) 
            A += dA * dt  # Note: multiply by dt for numerical integration
        kb = [48, 72, 24*6, 24*7]
        if k in kb:
            B += 6
        else:
            dB = antibiotic_decay(B, t_half_CAS)
            B += dB * dt
    

        pop_size = y0[0]
        mut_rate = 1e-8
        mut_prob = mut_rate * pop_size
        if np.random.random() < mut_prob:
                y0[1] += 1.01
        if np.random.random() < mut_prob:
                y0[2] += 1.01
        ts = [t[k-1], t[k]]
        y = odeint(model, y0, ts, args=(params, A, B, e_max))
        if y[1][1] < 1:
            y[1][1] = 0
        if y[1][0] < 1:
            y[1][0] = 0
        y0 = y[1]
        solution.append(y0)
        A_values.append(A)
        B_values.append(B)

    solution = np.array(solution)
    A_values = np.array(A_values)
    B_values = np.array(B_values)

    solution[solution < 1] = 0
    list_of_solutions.append(solution)


# print (list_of_solutions) to .csv file 

# Initialize an empty DataFrame for CSV data
csv_data = pd.DataFrame(columns=["Simulation_ID", "Wild_type", "resistant_to_AMB", "resistant_to_CAS"])


for i, final_state in enumerate(list_of_solutions):
    simulation_id = i
    wild_type = final_state[-1][0]
    resistant_AMB = final_state[-1][1]
    resistant_CAS = final_state[-1][2]

    csv_data.loc[i] = [simulation_id, wild_type, resistant_AMB, resistant_CAS]


csv_data.to_csv('cs_2day_multi_sim.csv', mode='w', index=False)


# pick only one solution from the list of solutions
solution = list_of_solutions[0]

print (solution)

# put solution and antibiotic data into a .csv file with column time, S, RA, RB, A, B, E
# Initialize an empty DataFrame for CSV data
csv_data = pd.DataFrame(columns=["time", "S", "RAMB", "RCAS", "A", "B", "E"])

for i, state in enumerate(solution):
    time = i
    S = state[0]
    RA = state[1]
    RB = state[2]
    A = A_values[i]
    B = B_values[i]
    E = state[4]

    csv_data.loc[i] = [time, S, RA, RB, A, B, E]


csv_data.to_csv('cs_2day.csv', mode='w', index=False)


# Before creating your plots, adjust the default plot settings for a larger display
plt.rcParams['axes.labelsize'] = 14  # Increase font size for axis labels
plt.rcParams['axes.titlesize'] = 16  # Increase font size for the plot title
plt.rcParams['xtick.labelsize'] = 12  # Increase font size for the x-axis tick labels
plt.rcParams['ytick.labelsize'] = 12  # Increase font size for the y-axis tick labels
plt.rcParams['legend.fontsize'] = 12  # Increase font size for legends
plt.rcParams['lines.linewidth'] = 3  # Increase the width of plotted lines
plt.rcParams['axes.grid'] = False # Enable grid by default for better readability

# Proceed with creating the combined plot with modifications for a larger, more compact display

fig, ax = plt.subplots(2, 1, figsize=(16, 12), sharex=True, gridspec_kw={'hspace': 0.1})


# Plot with increased line width and using the label font size
ax[0].plot(t/24, solution[:, 0], 'b-', label='WT', linewidth=3.5)  # Adjust linewidth as needed
ax[0].plot(t/24, solution[:, 1], 'r-', label='R_AMB', linewidth=3.5)
ax[0].plot(t/24, solution[:, 2], 'm-', label='R_CAS', linewidth=3.5)
ax[0].plot(t/24, solution[:, 4], 'k-', label='E Immune', linewidth=3.5)
ax[0].set_yscale('log')
# set lim for y axis
ax[0].set_ylim(0, 10**8)

# x and y ticks font size
ax[0].tick_params(axis='both', which='major', labelsize=16)
ax[1].tick_params(axis='both', which='major', labelsize=16)


# Customize legend and grid
ax[0].legend(loc='best')
#ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)
ax[0].grid(True, which='major', linestyle='--', linewidth=0.5)

# For the second subplot, similarly increase line width
ax[1].plot(t/24, A_values, 'g-', label='AMB', linewidth=3)
ax[1].plot(t/24, B_values, 'y-', label='CAS', linewidth=3)

# Adjust legend and grid for the second plot
ax[1].legend(loc='best')
ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)

plt.savefig('vivo_cs_2_day.png')