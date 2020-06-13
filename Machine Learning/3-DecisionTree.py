
#Data Import
import pandas as pd
dataset=pd.read_csv('C:/Users/HP/Desktop/SmartHomeData.csv', delimiter=';')
import numpy as np
Use=dataset['use [kW]']
Use=np.array(Use).reshape(-1,1)
Data=pd.read_csv('C:/Users/HP/Desktop/SmartHomeData.csv', delimiter=';')

#Tme Cleaning
time = pd.date_range('2019-01-01 05:00', periods=len(Data),  freq='min')  
time = pd.DatetimeIndex(time)
Data['time']=time

#Date & Hour
Data['day']= Data['time'].dt.day
Data['month']= Data['time'].dt.month
Data['week']= Data['time'].dt.week
Data['hour']= Data['time'].dt.hour

#MAPE Define
from sklearn.utils import check_arrays
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = check_arrays(y_true, y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


"""1/3-2/3 division"""
from sklearn.model_selection import train_test_split
x_train1,x_test1,train_label,test_label=train_test_split(Data, Use, test_size=0.33,random_state=0)

"""Decision Tree Regressor"""
from sklearn.tree import DecisionTreeRegressor
##train
mod = DecisionTreeRegressor()
mod= mod.fit(x_train1, train_label)
##predict
pred=mod.predict(x_test1)

  ##RMSE 
from sklearn.metrics import mean_squared_error
RMSE=np.sqrt(mean_squared_error(mod.predict(x_test1),test_label))
print("RMSE",RMSE)
print("MSE",mean_squared_error(mod.predict(x_test1),test_label))

##Explained Variance
from sklearn.metrics import explained_variance_score
EVS=explained_variance_score( test_label,mod.predict(x_test1))
print("EV" ,EVS)


#  ##MAPE 
mod.score(x_test1,test_label)
print("MAPE",mean_absolute_percentage_error(mod.predict(x_test1),test_label))

