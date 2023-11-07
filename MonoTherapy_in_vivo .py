# Importing necessary libraries and setting up the initial data and functions
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd


def antibiotic_decay(A, t_half):
    ke = np.log(2) / t_half
    return -ke * A

def model(y, t, params, f_A, f_C, e_max):
    S, RA, RB, A, E = y
    A = f_A
    B = f_C
    Bmax, KG, FIT, V, Gmax, Gmin_A, HILL_A, HILL_B,  MIC_S, CS_A, CS_B = params

    q = 10**-3
    l = 10**-3
    jN = 10**-6

    dSdt = S*(1-((S+RA+RB)/10**Bmax))*KG*(1 - ((Gmax - Gmin_A/KG)*(A/MIC_S)**HILL_A/((A/MIC_S)**HILL_A - (Gmin_A/KG/Gmax))) - ((Gmax - Gmin_B/KG)*(B/MIC_S)**HILL_B/((B/MIC_S)**HILL_B - (Gmin_B/KG/Gmax))))  - jN*S*E
    dRAdt = RA*(1-((S+RA+RB)/10**Bmax))*KG*FIT*(1- ((Gmax - Gmin_A/(KG*FIT))*(A/4)**HILL_A/((A/4)**HILL_A - (Gmin_A/(KG*FIT)/Gmax))) - ((Gmax - Gmin_B/(KG*FIT))*(B/(MIC_S*CS_B))**HILL_B/((B/(MIC_S*CS_B))**HILL_B - (Gmin_B/(KG*FIT)/Gmax)))) - jN*RA*E
    dRBdt = RB*(1-((S+RA+RB)/10**Bmax))*KG*(1- ((Gmax - Gmin_A/(KG*FIT))*(A/(MIC_S*CS_A))**HILL_A/((A/(MIC_S*CS_A))**HILL_A - (Gmin_A/(KG*FIT)/Gmax))) - ((Gmax - Gmin_B/KG)*(B/64)**HILL_B/((B/64)**HILL_B - (Gmin_B/KG/Gmax)))) - jN*RB*E
    
    dAdt = f_A
    dEdt = q*(e_max - E) - l*E

    return [dSdt, dRAdt, dRBdt, dAdt, dEdt]


Bmax = 9
KG = 0.21
FIT = 0.75
ke_bac = 0
V = 100
Gmax = 0.65
Gmin_A = -1
Gmin_B = -0.5
HILL_A = 4
HILL_B = 1
HILL_B = 0.5
MIC_S = 1
S0 = 10**7
RA0 = 0
RB0 = 0
A0 = 1.5
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

params = [Bmax, KG, 1,  V, Gmax, Gmin_A, HILL_A, HILL_B,  MIC_S, CS_A, CS_B]

A = 0
B = B0
t_half_AMB = 24
t_half_CAS = 6

e_max = 10**6

list_of_solutions=[]

for s in range(1, 100):
    solution = [y0]
    A_values = [A0]
    B_values = [B0]
    y0 = [S0, RA0,RB0, 1.5, 10**2]
    A = 1.5
    B = 0
    for k in range(1, length_experiment):
        # two days cycle

        if k % 24 == 0:
            A += A0  # Application of antibiotic A
        else:
            dA = antibiotic_decay(A, t_half_AMB) 
            A += dA * dt  # Note: multiply by dt for numerical integration
 # Note: multiply by dt for numerical integration


        pop_size = y0[0] 
        mut_rate = 10e-8
        mut_prob = mut_rate * pop_size
        if np.random.random() < mut_prob:
            if np.random.random() < 0.5:
                y0[1] += 1
            else:
                y0[2] += 1
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


csv_data.to_csv('mono_multi_1day.csv', mode='a', index=False)


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


csv_data.to_csv('dynamic_mono_AMB.csv', mode='a', index=False)


# make 2 subplots and share the x axis
fig, ax = plt.subplots(2, 1, figsize=(16, 12), sharex=True)

# plot the x axis time and y axis is the population of bacteria
ax[0].plot(t, solution[:, 0], 'b-', label='WT')
ax[0].plot(t, solution[:, 1], 'r-', label='R_AMB')
ax[0].plot(t, solution[:, 2], 'm-', label='R_CAS')
ax[0].plot(t, solution[:, 4], 'k-', label='E Immune')

#plot in log scale
ax[0].set_yscale('log')


# add a legend, and gridlines
ax[0].legend(loc='best')
ax[0].grid()



# in the subplot, plot the antibiotic concentration
ax[1].plot(t, A_values, 'g-', label='AMB')
ax[1].plot(t, B_values, 'y-', label='CAS')

# add a legend, and gridlines
ax[1].legend(loc='best')
ax[1].grid()



plt.savefig('vivo_mono_1_day.png')
