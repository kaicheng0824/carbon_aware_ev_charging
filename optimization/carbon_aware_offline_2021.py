#from typing_extensions import Required
import numpy as np
import csv
import cvxpy as cp #convex optimization
import matplotlib.pyplot as plt 
import argparse
import time
import data_handler_yearly
import csv
import pandas as pd

np.random.seed(0)
time_step = 5 # min
num_steps = int(24*(60/time_step)-1) # steps inn 24 hours
total_vehicles = 10
battery_capacity = 50 # kwh
power_capacity = 120 / 8 # kw, max power delivery every time step ~24 cars at same time
max_power_u = 5 / 8 # max power intake for cars every time step, ~6hours full charge
balancing_fac = 1
total_num_of_data_entries = 6742

arrival_time = np.zeros(total_num_of_data_entries)
departure_time = np.zeros(total_num_of_data_entries)
required_energy = np.zeros(total_num_of_data_entries)
initial_state= np.random.uniform(2.0, 20.0, size=(total_num_of_data_entries,)) # 4% - 40% SoC initial
final_energy = np.zeros(total_num_of_data_entries)
carbon_intensity = np.zeros(num_steps)

# Get Berkley Data
arrival_time, departure_time, required_energy, date= data_handler_yearly.getCleanedBerkleyData(arrival_time, departure_time, required_energy)
#print(required_energy)
#print(len(np.where(np.array(initial_state + required_energy) > battery_capacity * np.ones(total_num_of_data_entries,))[0]))
final_energy = np.minimum(np.array(initial_state + required_energy), battery_capacity * np.ones(total_num_of_data_entries,))
#print('final',final_energy, 'initial', initial_state, 'required', required_energy)
#print("final_shape", np.shape(final_energy))
#print('maxfinal',max(final_energy))
#print('date',date)
#arrival_time = [int(i*12) for i in arrival_time]
#departure_time = [int(i*12) for i in departure_time]
#print(np.shape(arrival_time))
#print(departure_time)

# Get Carbon Intensity Data
carbon_intensity = np.array(data_handler_yearly.getCarbonIntensityData1year(),dtype=float)
carbon_intensity = carbon_intensity[:, 1:]
print("carbon intensity data shape", np.shape(carbon_intensity))

def carbon_aware_MPC(carbon_intensity, num_of_vehicles, timesteps,
                     initial_states, max_power, terminal_states,
                     arrival_time, dept_time, power_capacity,
                     B, factor, day):
    x_terminal=cp.Parameter(num_of_vehicles, name='x_terminal') # Requested end SoC for all vehicles
    x0 = cp.Parameter(num_of_vehicles, name='x0') # Initial SoC for all vehicles
    max_sum_u = cp.Parameter(name='max_sum_u') # Peak charging power for the infrastructure
    u_max = cp.Parameter(name='u_max') # Maximum charging power for each vehicle at each time step
    
    x = cp.Variable((num_of_vehicles, timesteps+1), name='x') # SoC at each time step for each vehicle
    u = cp.Variable((num_of_vehicles, timesteps), name='u') # charging power at each time step for each vehicle

    #print(terminal_states)
    x_terminal.value = terminal_states
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
        obj += cp.sum(u[:,t]*carbon_intensity[day, t])

    obj += factor * cp.norm(x[:, -1] - x_terminal, 2)
    
    # Solve Convex Optimization Here
    prob = cp.Problem(cp.Minimize(obj), constr)
    prob.solve()

    #print("battery state", x.value)
    #plt.plot(x.value[0]/B,label='SoC')
    #plt.plot(x.value[1]/B,label='SoC')
    #plt.plot(u.value,label='charging_power')
    #plt.plot(np.array(carbon_intensity[day,:], dtype=float),'g',label='carbon_intensity')
    #print(f'arrival time:{arrival_time[1]}')
    #plt.legend()
    #plt.show()

    return x.value/B, u.value




#path = '../result/MPC_offline_P30.csv'
for balancing_fac in [1]:
    starting_index=0
    end_index=0
    for current_date in range(365)
        starting_index=np.copy(end_index)
        num_of_vehicles=0

        while date[end_index]==date[starting_index]:
            num_of_vehicles+=1
            end_index+=1
        ini_state=initial_state[starting_index:end_index]
        final_state=final_energy[starting_index:end_index]
        arr_time=arrival_time[starting_index:end_index]
        dept_time=departure_time[starting_index:end_index]

        print("number of cars", num_of_vehicles)

        x, u = carbon_aware_MPC(carbon_intensity, num_of_vehicles, num_steps,
                                ini_state, max_power_u, final_state,
                                arr_time,  dept_time, power_capacity,
                                battery_capacity, factor=balancing_fac, day=current_date)
        carbon_emission = np.sum(np.array([u[:,t]*carbon_intensity[current_date, t] for t in range(num_steps)]))
        print(f'the energy delivery: {round(np.sum(u),2)}, '
            f'the required energy: {round(np.sum(required_energy[starting_index:end_index]),2)}, '
            f'the carbon emission term: {round(carbon_emission,2)}')
        required = round(np.sum(required_energy[starting_index:end_index]),2)
        delivery = round(np.sum(u),2)
        # with open(path,'a+') as f:
        #     csv_write = csv.writer(f)
        #     data = ['0',current_date,balancing_fac,required,delivery,round(delivery/required*100,2), end_index-starting_index, carbon_emission*0.907]
        #     csv_write.writerow(data)
        np.savez(f'../result/season/offline_day{current_date}_P120.npz',x=x,u=u,carbon=np.array([u[:,t]*carbon_intensity[current_date, t] for t in range(num_steps)]))

       