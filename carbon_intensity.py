import csv
import numpy as np
import csv_handler

data_points = 24*12

co2_path = 'CAISO_2021_CO2.csv'
supply_path = 'CAISO_2021_supply.csv'

row_count_co2 = 7
row_count_supply = 9

def read_sum(path, row_count, tot_sum): 
    with open(path, 'r') as file:
            reader = csv.reader(file)
            next(reader)
            
            line_count = 0
            for row in reader:
                if(line_count==row_count-1):
                    theRow = list(row)
                    tot_sum = theRow[1:]
                    tot_sum = [float(i) for i in tot_sum]
                    
                else: 
                    line_count += 1
                    continue

    return tot_sum

def getFields(fields,path):
    with open(path, 'r') as file:
        reader = csv.reader(file)
        
        k = 0
        for row in reader:
            fields = np.array(list(row))
            k = k+1
            if(k>=1):
                break

    return fields

def write_csv(name,tot_sum,fields,row_count):
    with open(name, 'w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(fields)
        writer.writerow(tot_sum)

def get_carbon_intensity():
    return [float(float(i) / float(j)) for i, j in zip(tot_sum_co2, tot_sum_supply)]

if __name__ == "__main__":
    tot_sum_co2 = np.zeros((1,data_points), dtype=float)
    tot_sum_supply = np.zeros((1,data_points), dtype=float)
    carbon_intensity = np.zeros((1,data_points), dtype=float)
    fields = np.zeros(data_points, dtype=object)

    tot_sum_supply = read_sum(supply_path, row_count_supply, tot_sum_supply)
    tot_sum_co2 = read_sum(co2_path, row_count_co2, tot_sum_co2)

    carbon_intensity = get_carbon_intensity()
    fields = getFields(fields,supply_path)
    row_count = 2
    write_csv('CAISO_2021_CarbonIntensity.csv', carbon_intensity,fields,row_count)
