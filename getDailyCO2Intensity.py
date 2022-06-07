import csv
import numpy as np
import calculation
import csv_handler

# Parse through all CO2 data in 2021 and sum them up
co2_row_count = 7
supply_row_count = 9
data_points = 24*12
co2_all_sum = np.zeros((co2_row_count,data_points), dtype=int)
supply_all_sum  = np.zeros((supply_row_count,data_points), dtype=int)
fields = np.zeros(data_points, dtype=object)

def createPath():
    big = 1
    small = 2
    smallsmall = 3
    months = [big,smallsmall,big,small,big,small,big,big,small,big,small,big]
    month = np.array(months)

    co2_path = np.zeros((1,365),dtype='|S100')
    supply_path = np.zeros((1,365),dtype='|S100')

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
        
            co2_path[0,dayCount] = 'data/CO2/CAISO_CO2_per_resource_2021/CAISO-co2-per-resource-2021{}.csv'.format(theMonth_str + theDay_str)
            supply_path[0,dayCount] = 'data/Supply/CAISO_supply_2021/CAISO-supply-2021{}.csv'.format(theMonth_str + theDay_str)
        
            dayCount = dayCount+1
    
        theMonth = theMonth+1
    return co2_path, supply_path

co2_path, supply_path = createPath() #Get CO2 File Path
dailyco2Matrix = np.zeros((366, data_points),dtype=object)
dailySupplyMatrix = np.zeros((366, data_points),dtype=object)
dailyIntensityMatrix = np.zeros((366, data_points),dtype=object)

for a in range(365):
    with open(co2_path[0][a], 'r') as file:
        reader = csv.reader(file)
        next(reader)

        i = 0
        for row in reader:
            theList = list(row)

            for j in range(data_points):
                if(theList[j+1]==''):
                    continue

                co2_all_sum[i,j] = theList[j+1]

            i = i + 1

            if(i < co2_row_count):
                continue
            else:
                break

    for k in range(co2_row_count):
        co2_all_sum[co2_row_count-1,1:] = co2_all_sum[co2_row_count-1,1:] + co2_all_sum[k,1:]

    dailyco2Matrix[a,1:] = co2_all_sum[6,1:]
    co2_all_sum = np.zeros((co2_row_count,data_points), dtype=int)

    with open(supply_path[0][a], 'r') as file:
        reader = csv.reader(file)
        next(reader)

        i = 0
        for row in reader:
            theList = list(row)

            for j in range(data_points):
                if(theList[j+1]==''):
                    continue

                supply_all_sum[i,j] = theList[j+1]

            i = i + 1

            if(i < supply_row_count):
                continue
            else:
                break

    for k in range(supply_row_count):
        supply_all_sum[supply_row_count-1,1:] = supply_all_sum[supply_row_count-1,1:] + supply_all_sum[k,1:]

    #print(supply_all_sum[8])
    dailySupplyMatrix[a,1:] = supply_all_sum[supply_row_count-1,1:]
    supply_all_sum = np.zeros((supply_row_count,data_points), dtype=int)

#-----------------------------------------------------#
dailyco2Matrix = dailyco2Matrix.flatten()
dailySupplyMatrix = dailySupplyMatrix.flatten()
dailyIntensityMatrix = dailyIntensityMatrix.flatten()

for i in range(366*data_points):
    if (dailySupplyMatrix[i] == 0):
        continue
    dailyIntensityMatrix[i] = dailyco2Matrix[i] / dailySupplyMatrix[i]

dailyIntensityMatrix = dailyIntensityMatrix.reshape(366,data_points)

def getFields(fields):
    with open(co2_path[0,0], 'r') as file:
        reader = csv.reader(file)
        
        k = 0
        for row in reader:
            fields = np.array(list(row))
            k = k+1
            if(k>=1):
                break
    
    return fields

fields = getFields(fields)

def createDate():
    big = 1
    small = 2
    smallsmall = 3
    months = [big,smallsmall,big,small,big,small,big,big,small,big,small,big]
    month = np.array(months)

    date = np.zeros((1,365),dtype='|S100')

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
        
            date[0,dayCount] = '2021{}'.format(theMonth_str + theDay_str)
            dayCount = dayCount+1
    
        theMonth = theMonth+1
    return date

# Add Date
date = createDate()

for i in range(365):
    dailyIntensityMatrix[i][0] = date[0][i].decode("utf-8")

csv_handler.write_csv('CAISO_2021_Daily_CO2_Intensity.csv', dailyIntensityMatrix,fields,365)