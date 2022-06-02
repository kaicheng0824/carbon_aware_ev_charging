import csv

def getBerkleyData(arrival_time,departure_time,required_energy):
    with open('data/Berkley EV Charging/LBNL_Data.csv', newline='') as file:
        reader = csv.reader(file)
        next(reader)

        line_count = 0
        for row in reader:
            if(line_count == 10):
                break
            else:
                theRow = list(row)
                arrival_time[line_count] = (float(theRow[2][11:13]) + float(theRow[2][14:16])/60)
                departure_time[line_count] = (float(theRow[3][11:13]) + float(theRow[3][14:16])/60)
                required_energy[line_count] = (float(theRow[6]))
                line_count += 1

    return arrival_time, departure_time, required_energy

def getCarbonIntensityData(carbon_intensity):
    with open('CAISO_2021_CarbonIntensity.csv', newline='') as file:
        reader = csv.reader(file)
        next(reader)

        for row in reader:
            theRow = list(row)
            carbon_intensity = theRow

    return carbon_intensity