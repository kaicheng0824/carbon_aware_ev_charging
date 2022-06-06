import csv
import numpy as np
import csv_handler

arrival_time = np.zeros(10000,dtype=object)
departure_time = np.zeros(10000,dtype=object)
date = np.zeros(10000,dtype=object)
hashcode = np.zeros(10000,dtype=object)
required_energy = np.zeros(10000,dtype=object)

with open('data/Berkley_EV_Charging/LBNL_Data.csv', newline='') as file:
    reader = csv.reader(file)
    next(reader)

    line_count = 0
    for row in reader:
        theRow = list(row)
        #print(theRow[2][0:4])
        if(theRow[2][0:4]!=str(2021)):
            continue
        else:  
            arrival_time[line_count] = theRow[2][11:13]+':'+theRow[2][14:16]
            departure_time[line_count] = theRow[3][11:13]+':'+theRow[3][14:16]
            required_energy[line_count] = (float(theRow[6]))
            hashcode[line_count] = theRow[0]
            date[line_count] = theRow[2][0:4] + theRow[2][5:7] + theRow[2][8:10]

            line_count += 1

arrival_time = arrival_time[0:line_count]
departure_time = departure_time[0:line_count]
required_energy = required_energy[0:line_count]
hashcode = hashcode[0:line_count]
date = date[0:line_count]

# print(arrival_time)
# print(departure_time)
# print(required_energy)
# print(hashcode)
# print(date)

summary = np.vstack((date,hashcode,arrival_time,departure_time,required_energy)).T

fields = np.array((['date','hashcode','arrival time', 'departure time', 'required energy']))
csv_handler.write_csv('BerkeleyDataCleaned.csv',summary,fields,len(date)-1)