import csv

def write_csv(name,tot_sum,fields,row_count):
    with open(name, 'w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(fields)

        for i in range(row_count):
            writer.writerow(tot_sum[i].tolist())