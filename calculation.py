import csv

def sum(days, path, data_points, all_sum, tot_sum, row_count):
    for a in range(days):
        if(path[a]==b''):
            continue

        with open(path[a], 'r') as file:
            reader = csv.reader(file)
            next(reader)

            i = 0
            for row in reader:
                theList = list(row)

                for j in range(data_points):
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

    return tot_sum