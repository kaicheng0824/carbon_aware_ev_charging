import numpy as np
import csv
import cvxpy as cp #convex optimization
import matplotlib.pyplot as plt 
import argparse
import time
import data_handler

time_step = 5 # min
num_steps = int(24*(60/time_step)-1) # steps inn 24 hours
total_vehicles = 10
battery_capacity = 50 # kwh
power_capacity = 120 # kw, max power delivery 
max_power_u = 3 # max power intake for cars

arrival_time = np.zeros(total_vehicles)
departure_time = np.zeros(total_vehicles)
required_energy = np.zeros(total_vehicles)
initial_state= np.random.uniform(0.8, 4.0, size=(total_vehicles,))
final_energy = np.zeros(total_vehicles)
carbon_intensity = np.zeros(num_steps)

# Get Berkley Data
arrival_time,
departure_time, 
required_energy = data_handler.getBerkleyData(arrival_time,departure_time, required_energy)
final_energy = initial_state + required_energy

#print(arrival_time)
arrival_time = [int(i*12) for i in arrival_time]
#print(arrival_time)

# Get Carbon Intensity Data
carbon_intensity = data_handler.getCarbonIntensityData(carbon_intensity)

def carbon_aware_MPC(carbon_intensity, num_of_vehicles, timesteps, initial_states, max_power, terminal_states, arrival_time, dept_time, power_capacity, B, factor = 1):
    x_terminal=cp.Parameter(num_of_vehicles, name='x_terminal') # Requested end SoC for all vehicles
    x0 = cp.Parameter(num_of_vehicles, name='x0') # Initial SoC for all vehicles
    max_sum_u = cp.Parameter(name='max_sum_u') # Peak charging power for the infrastructure
    u_max = cp.Parameter(name='u_max') # Maximum charging power for each vehicle at each time step
    
    x = cp.Variable((num_of_vehicles, timesteps+1), name='x') # SoC at each time step for each vehicle
    u = cp.Variable((num_of_vehicles, timesteps), name='u') # charging power at each time step for each vehicle

    #print(terminal_states)
    x_terminal.value=terminal_states[0][:]
    x0.value = initial_states
    max_sum_u.value = power_capacity
    u_max.value = max_power

    constr = [x[:,0] == x0,  x[:,-1] <= x_terminal]

    for i in range(num_of_vehicles):
        constr += [x[i,1:] == x[i,0:timesteps] + u[i,:], u[i,:] >= 0,]
        for t in range(timesteps):
            constr += [u[i, t] <= u_max*(t>=arrival_time[i]),
                       u[i, t] <= u_max*(t<dept_time[i])]
    
    obj = 0.
    for t in range (timesteps):
        constr += [cp.sum(u[0:num_of_vehicles,t]) <= power_capacity]
        obj += cp.sum(u[:,t]*carbon_intensity[t])

    obj += factor * cp.norm(x[:, -1] - x_terminal, 2)
    
    # Solve Convex Optimization Here
    prob = cp.Problem(cp.Minimize(obj), constr)
    prob.solve()

    # Plotting the 10th Car
    plt.plot(x.value[5]/B,'r',label='SoC')
    plt.plot(u.value[5],'b',label='charging_power')
    #print(np.array(carbon_intensity,dtype=float))
    plt.plot(np.array(carbon_intensity,dtype=float),'g',label='carbon_intensity')
    print(f'arrival time:{arrival_time[5]}')
    plt.show()

    return x.value/B, u.value

print(required_energy)
x, u = carbon_aware_MPC(carbon_intensity, total_vehicles, num_steps, initial_state, max_power_u, final_energy, arrival_time,  departure_time, power_capacity, battery_capacity, factor=60)
print(f'the energy delivery: {round(np.sum(u),2)}, the required energy: {round(np.sum(required_energy),2)}')
carbon_emission = np.sum(np.array([u[:,t]*carbon_intensity[t] for t in range(num_steps)]))
print(f'the energy delivery: {round(np.sum(u),2)}, the required energy: {round(np.sum(required_energy),2)}, the carbon emission term: {round(carbon_emission,2)}')

