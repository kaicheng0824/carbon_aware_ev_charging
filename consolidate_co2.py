import csv
import numpy as np
import calculation
import csv_handler

# Parse through all CO2 data in 2021 and sum them up

row_count = 7
data_points = 24*12
natural_gas_sum = np.array(data_points)
import_gas_sum = np.array(data_points)
all_sum = np.zeros((row_count,data_points), dtype=int)
fields = np.zeros(data_points, dtype=object)

def createPath():
    big = 1
    small = 2
    smallsmall = 3
    months = [big,smallsmall,big,small,big,small,big,big,small,big,small,big]
    month = np.array(months)

    path = np.zeros((1,366),dtype='|S100')

    theMonth = 1
    dayCount = 0
    for month in months:
        if(month==big):
            maxDay = 31
        elif(month==small):
            maxDay = 30
        else:
            maxDay = 28
    
        for day in range(maxDay):
        # Adjust day to normal 1 to 30
            day = day + 1
        #print(day)
        
            if(theMonth < 10):
                theMonth_str = str(theMonth).zfill(2)
            else:
                theMonth_str = str(theMonth)

            if(day < 10):
                theDay_str = str(day).zfill(2)
            else:
                theDay_str = str(day)
        
            path[0,dayCount] = 'data/CO2/CAISO_CO2_per_resource_2021/CAISO-co2-per-resource-2021{}.csv'.format(theMonth_str + theDay_str)
        
            dayCount = dayCount+1
    
        theMonth = theMonth+1
    return path

path = createPath()

# Daylight saving started on March 14th (73nd day) to November 7th (311th day)
path_daylight_saving = path[0,72:311]
path_non_daylight = np.concatenate((path[0,311:-1],path[0,0:72]), axis=None)
all_sum_daylight = np.zeros((row_count,data_points), dtype=int)
all_sum_non_daylight = np.zeros((row_count,data_points), dtype=int)
tot_sum_daylight = np.zeros((row_count,data_points), dtype=object)
tot_sum_non_daylight = np.zeros((row_count,data_points), dtype=object)

tot_sum = np.zeros((row_count,data_points), dtype=object)


tot_sum = calculation.sum(365,path[0,:],data_points,all_sum,tot_sum,row_count)
tot_sum_non_daylight = calculation.sum(365-239,path_non_daylight
                        ,data_points
                        ,all_sum_non_daylight
                        ,tot_sum_non_daylight
                        ,row_count)

tot_sum_daylight = calculation.sum(238,path_daylight_saving
                        ,data_points
                        ,all_sum_daylight
                        ,tot_sum_daylight
                        ,row_count)

with open(path[0,0], 'r') as file:
        reader = csv.reader(file)
        next(reader)
        
        k = 0
        for row in reader:
            tot_sum[k,0] = list(row)[0]
            tot_sum_daylight[k,0] = list(row)[0]
            tot_sum_non_daylight[k,0] = list(row)[0]
            k = k + 1

def getFields(fields):
    with open(path[0,0], 'r') as file:
        reader = csv.reader(file)
        
        k = 0
        for row in reader:
            fields = np.array(list(row))
            k = k+1
            if(k>=1):
                break
    
    return fields

fields = getFields(fields)

# Summing all different sources
for i in range(data_points-1):
    for j in range(row_count-1):
        tot_sum[6,i+1] = tot_sum[6,i+1] + tot_sum[j,i+1]
        tot_sum_daylight[6,i+1] = tot_sum_daylight[6,i+1] + tot_sum_daylight[j,i+1]
        tot_sum_non_daylight[6,i+1] = tot_sum_non_daylight[6,i+1] + tot_sum_non_daylight[j,i+1]

tot_sum[6,0] = 'Sum'
tot_sum_daylight[6,0] = 'Sum'
tot_sum_non_daylight[6,0] = 'Sum'

# Normalize to 365 days and twelve 5 minute session (or 238 for daylight saving)
for i in range(row_count):
    tot_sum[i,1:] = tot_sum[i,1:]/365/12
    tot_sum_daylight[i,1:] = tot_sum_daylight[i,1:]/238/12
    tot_sum_non_daylight[i,1:] = tot_sum_non_daylight[i,1:]/(365-238)/12

csv_handler.write_csv('CAISO_2021_CO2_daylight.csv', tot_sum_daylight,fields,row_count)
csv_handler.write_csv('CAISO_2021_CO2.csv', tot_sum,fields,row_count)
csv_handler.write_csv('CAISO_2021_CO2_non_daylight.csv', tot_sum_non_daylight,fields,row_count)

