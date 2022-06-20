import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import data_handler_yearly
import data_handler_yearly

def carbon_intensity_forecast():
    carbon_intensity = np.array(data_handler_yearly.getCarbonIntensityData1year(),dtype=float)
    carbon_intensity = carbon_intensity[:, 1:]
    print(np.shape(carbon_intensity))
    carbon_all=carbon_intensity.reshape(-1,1)
    #plt.plot(carbon_all[287*15:287*30])
    #plt.show()

    load_data=data_handler_yearly.getLoadCAISO()
    print(np.shape(load_data))

    weekday_index=6
    hour_index=0
    month_index=0
    minute_index=0
    feature_mat=np.zeros((carbon_all.shape[0]-24*12, 8), dtype=float)
    start_index=24*12
    monthday_index=1
    day_index=1

    for i in range(carbon_all.shape[0]-24*12):
        feature_mat[i,0]=np.average(carbon_all[start_index-24*12:start_index]) #24-hour average
        feature_mat[i,1]=np.average(carbon_all[start_index-12*12:start_index]) #12-hour average
        feature_mat[i,2]=load_data[day_index*24+hour_index]
        feature_mat[i,3]=weekday_index
        feature_mat[i,4]=hour_index
        feature_mat[i,5]=minute_index
        feature_mat[i,6]=month_index
        feature_mat[i,7]=np.average(carbon_all[start_index-12:start_index]) #1-hour average


        minute_index+=1
        start_index+=1


        if minute_index==12:
            minute_index=0
            hour_index += 1
            if hour_index == 24:
                hour_index = 0
                day_index += 1
                weekday_index += 1
                monthday_index+=1
                if weekday_index == 7:
                    weekday_index = 0
                if monthday_index==30:
                    monthday_index=0
                    month_index+=1

    #plt.plot(feature_mat[:288*300,6])
    #plt.show()
    start_index=24*12
    Y=carbon_all[start_index:]
    Y_test=Y[:288*4]

    reg = LinearRegression().fit(feature_mat, Y)
    return feature_mat, reg

#y_pred=reg.predict(feature_mat[:288*4])
#plt.plot(y_pred,c='b',label='predicted')
#plt.plot(Y[:288*4], c='g',label='Ground Truth')
#plt.legend()
#plt.show()