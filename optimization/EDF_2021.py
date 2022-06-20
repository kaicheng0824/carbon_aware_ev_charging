import numpy as np
import csv
import cvxpy as cp  # convex optimization
import matplotlib.pyplot as plt
import argparse
import time
import data_handler_yearly
from carbon_forecasts import carbon_intensity_forecast
parser = argparse.ArgumentParser()
parser.add_argument("--P", type=float, default=120, help="maxP")

opts = parser.parse_args()

np.random.seed(0)
time_step = 5  # min
num_steps = int(24 * (60 / time_step) - 1)  # steps inn 24 hours
total_vehicles = 10
battery_capacity = 50  # kwh
power_capacity = opts.P / 8  # kw, max power delivery every time step ~24 cars at same time
max_power_u = 5.0 / 8  # max power intake for cars every time step, ~6hours full charge
balancing_fac = 1
total_num_of_data_entries = 6742

arrival_time = np.zeros(total_num_of_data_entries)
departure_time = np.zeros(total_num_of_data_entries)
required_energy = np.zeros(total_num_of_data_entries)
initial_state = np.random.uniform(2.0, 20.0, size=(total_num_of_data_entries,))  # 4% - 40% SoC initial
final_energy = np.zeros(total_num_of_data_entries)
carbon_intensity = np.zeros(num_steps)

# Get Berkley Data
arrival_time, departure_time, required_energy, date = data_handler_yearly.getCleanedBerkleyData(arrival_time,
                                                                                                departure_time,
                                                                                                required_energy)
final_energy = np.minimum(np.array(initial_state + required_energy),
                          battery_capacity * np.ones(total_num_of_data_entries, ))
#print('final',final_energy, 'initial', initial_state, 'required', required_energy)
# print("final_shape", np.shape(final_energy))
# print('maxfinal',max(final_energy))
# print('date',date)
# arrival_time = [int(i*12) for i in arrival_time]
# departure_time = [int(i*12) for i in departure_time]
# print(np.shape(arrival_time))
# print(departure_time)

# Get Carbon Intensity Data
carbon_intensity = np.array(data_handler_yearly.getCarbonIntensityData1year(), dtype=float)
carbon_intensity = carbon_intensity[:, 1:]
print("carbon intensity data shape", np.shape(carbon_intensity))


def Earliest_deadline(num_of_vehicles, timesteps,
                     initial_states, max_power, terminal_states,
                     arrival_time, dept_time, power_capacity,
                     B, factor, day):

    initial_state_EDF = np.copy(initial_states)
    all_state_EDF = np.zeros((num_of_vehicles, timesteps+1))
    for t in range(timesteps):
        all_state_EDF[:,t] = initial_state_EDF[:]
    u_mat=np.zeros((num_of_vehicles, timesteps), dtype=float)

    #-5 to avoid computation infeasibility at this time
    for t in range(int(arrival_time[0])+1, timesteps-5):
        power_budget=power_capacity #Change this for variable case

        #print("Current time", t)

        #Firstly get the states
        #print("current number of arrived cars", (arrival_time < t).sum())
        vehicle_ending_index = (arrival_time < t).sum()
        step_initial_SOC = np.copy(initial_state_EDF[:vehicle_ending_index])
        depart_schedule=np.copy(dept_time[:vehicle_ending_index])
        u_val=np.zeros_like(step_initial_SOC) #record available cars charging rate
        index=np.argsort(depart_schedule) #sort the departure time
        charging_sessions=0

        while power_budget>=0:
            if depart_schedule[index[charging_sessions]] >= t: #not departed yet
                available_charging=np.minimum(max_power, power_budget)
                u_val[index[charging_sessions]] = np.maximum(np.minimum(available_charging, terminal_states[index[charging_sessions]]-step_initial_SOC[index[charging_sessions]]),0)

            power_budget -= u_val[index[charging_sessions]]
            charging_sessions+=1

            if charging_sessions>=vehicle_ending_index:
                break

        #print("SUM EDF", np.sum(u_val))
        updated_val = u_val
        #print("U after MPC cut", updated_val)
        #print("SUM EDF Cut", np.sum(updated_val))
        initial_state_EDF[:vehicle_ending_index] += updated_val
        #print("SOC_states", np.round(initial_state_SOC[:vehicle_ending_index],2))
        u_mat[:vehicle_ending_index, t]=updated_val
        all_state_EDF[:,t] = initial_state_EDF[:]

    # plt.plot(all_state_EDF[10,:-5]/B,label='SoC')
    # plt.plot(u_mat[10,:-5],label='charging_power')
    # plt.plot(carbon_intensity,label='carbon_intensity')
    # print(f'arrival time:{arrival_time[10]}')
    # plt.show()
    
    return all_state_EDF/B, u_mat



starting_index = 0
end_index = 0
path = f'../result/EDF_P{int(opts.P)}.csv'
feature_mat, carbon_model = carbon_intensity_forecast()

for current_date in range(1, 365):

    starting_index = np.copy(end_index)
    num_of_vehicles = 0

    while date[end_index] == date[starting_index]:
        num_of_vehicles += 1
        end_index += 1
    ini_state = initial_state[starting_index:end_index]
    final_state = final_energy[starting_index:end_index]
    arr_time = arrival_time[starting_index:end_index]
    dept_time = departure_time[starting_index:end_index]

    print("DAY ", current_date)
    print("number of cars", num_of_vehicles)

    #First implement offline algorithm
    x, u = Earliest_deadline(num_of_vehicles, num_steps,
                            ini_state, max_power_u, final_state,
                            arr_time, dept_time, power_capacity,
                            battery_capacity, factor=balancing_fac, day=current_date)
    carbon_emission = np.sum(np.array([u[:, t] * carbon_intensity[current_date, t] for t in range(num_steps)]))
    print(f'the energy delivery: {round(np.sum(u), 2)}, '
          f'the required energy: {round(np.sum(required_energy[starting_index:end_index]), 2)}, '
          f'the carbon emission term: {round(carbon_emission, 2)}')
    required = round(np.sum(required_energy[starting_index:end_index]),2)
    delivery = round(np.sum(u),2)
    with open(path,'a+') as f:
        csv_write = csv.writer(f)
        data = ['0',current_date,-1,required,delivery,round(delivery/required*100,2), end_index-starting_index, carbon_emission*0.907]
        csv_write.writerow(data)


