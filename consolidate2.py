import csv
import numpy as np

row_count = 9
data_points = 24*12
natural_gas_sum = np.array(data_points)
import_gas_sum = np.array(data_points)
all_sum = np.zeros((row_count,data_points), dtype=int)
fields = np.zeros(data_points, dtype=object)

big = 1
small = 2
smallsmall = 3
months = [big,smallsmall,big,small,big,small,big,big,small,big,small,big]
month = np.array(months)

path = np.zeros((1,366),dtype='|S40')
print(path.shape)

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
        
        path[0,dayCount] = 'CAISO-supply-2021{}.csv'.format(theMonth_str + theDay_str)
        
        dayCount = dayCount+1
    
    theMonth = theMonth+1

#print(path)
tot_sum = np.zeros((row_count,data_points), dtype=object)
#print(tot_sum)

for a in range(366):
    if(path[0,a]==b''):
        continue

    with open(path[0,a], 'r') as file:
        reader = csv.reader(file)
        next(reader)

        i = 0
        for row in reader:
            theList = list(row)
            #theList = theList[1:]
            #print(theList)

            for j in range(data_points):
                # print(i)
                # print('-----')
                # print(j)
                if(theList[j+1]==''):
                    continue

                all_sum[i,j] = theList[j+1]

            i = i + 1

            if(i < row_count):
                continue
            else:
                break

    for k in range(row_count):
        tot_sum[k,1:] = tot_sum[k,1:] + all_sum[k,1:]
        
with open(path[0,0], 'r') as file:
        reader = csv.reader(file)
        next(reader)
        
        k = 0
        for row in reader:
            tot_sum[k,0] = list(row)[0]
            k = k + 1

with open(path[0,0], 'r') as file:
        reader = csv.reader(file)
        
        k = 0
        for row in reader:
            fields = np.array(list(row))
            k = k+1
            if(k>=1):
                break

# Summing all different sources
for i in range(data_points-1):
    for j in range(row_count-1):
        tot_sum[8,i+1] = tot_sum[8,i+1] + tot_sum[j,i+1]

tot_sum[8,0] = 'Sum'

with open('CAISO_2021_supply.csv', 'w', newline='') as file:
    writer = csv.writer(file)

    writer.writerow(fields)

    for i in range(row_count):
        writer.writerow(tot_sum[i].tolist())






