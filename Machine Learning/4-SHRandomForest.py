
##Data import
import pandas as pd
Data=pd.read_csv('C:/Users/HP/Desktop/SmartHomeData.csv', delimiter=';')

#Data affection 
import numpy as np
Use=Data['use [kW]']
Use=np.array(Use).reshape(-1,1)

Used_Cols=['Home office [kW]','Wine cellar [kW]','Kitchen 12 [kW]','Kitchen 14 [kW]','Kitchen 38 [kW]','Barn [kW]','Living room [kW]','Dishwasher [kW]','Furnace 1 [kW]','Furnace 2 [kW]','Fridge [kW]','Garage door [kW]','Well [kW]','Microwave [kW]','Solar [kW]','time','temperature','apparent temperature']
Unused=['use [kW]','House overall [kW]','gen [kW]']
for i in Unused:
    Data=Data.drop(i,axis=1)
Data=np.array(Data)

"""1/3-2/3 division"""
from sklearn.model_selection import train_test_split
x_train1,x_test1,train_label,test_label=train_test_split(Data, Use, test_size=0.9,random_state=0)


"""Random Forest Regressor"""
from sklearn.ensemble import RandomForestRegressor
import time
depart=time.time()

regr = RandomForestRegressor(max_depth=20, random_state=0 ,n_estimators=100 )
regr.fit(x_train1, train_label)
predreg=regr.predict(x_test1)

temps=time.time()-depart

  ##R Squarred
Rsq=regr.score(x_test1,test_label)

 ##RMSE 
from sklearn.metrics import mean_squared_error
RMSE=np.sqrt(mean_squared_error(test_label, regr.predict(x_test1)))
print("RMSE",RMSE)

###Explained Variance

from sklearn.metrics import explained_variance_score
EVS=explained_variance_score(regr.predict(x_test1), test_label)
print("explained Variance" ,EVS)

###MAE
from sklearn.metrics import mean_absolute_error
print("MAE",mean_absolute_error(test_label,regr.predict(x_test1)))

