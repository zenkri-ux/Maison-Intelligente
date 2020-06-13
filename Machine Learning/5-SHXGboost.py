
##Data import
import xgboost as xgb
import pandas as pd
import numpy as np

Data=pd.read_csv('C:/Users/HP/Desktop/SmartHomeData.csv', delimiter=';')


#Data affection 
Use=Data['use [kW]']
Use=np.array(Use).reshape(-1,1)

Used_Cols=['Home office [kW]','Wine cellar [kW]','Kitchen 12 [kW]','Kitchen 14 [kW]','Kitchen 38 [kW]','Barn [kW]','Living room [kW]','Dishwasher [kW]','Furnace 1 [kW]','Furnace 2 [kW]','Fridge [kW]','Garage door [kW]','Well [kW]','Microwave [kW]','Solar [kW]','time','temperature','apparent temperature']
Unused=['use [kW]','House overall [kW]','gen [kW]']
for i in Unused:
    Data=Data.drop(i,axis=1)

Data=np.array(Data)



"""1/3-2/3 division"""
from sklearn.model_selection import train_test_split
x_train1,x_test1,train_label,test_label=train_test_split(Data, Use, test_size=0.33,random_state=0)

"""XGBoost Regressor"""
import time
depart=time.time()
# fit model w training data
model = xgb.XGBRegressor(n_estimators=50,max_depth=20, random_state=0,n_jobs=1)
model.fit(x_train1, train_label)

temps=time.time()-depart

# make predictions for test data
y_pred = model.predict(x_test1)

  ##R Squarred
Rsq=model.score(x_test1,test_label)

##RMSE
from sklearn.metrics import mean_squared_error
RMSE=np.sqrt(mean_squared_error(test_label, y_pred))
print("RMSE",RMSE)

 

###Explained Variance
from sklearn.metrics import explained_variance_score
EVS=explained_variance_score(model.predict(x_test1), test_label)
print("EV" ,EVS)

##MAE
from sklearn.metrics import mean_absolute_error
print("MAE",mean_absolute_error(test_label,model.predict(x_test1)))


import matplotlib.pyplot as plt 
plt.figure(figsize=(30,14))

plt.plot(model.predict(x_test1)[:500], label='Predicted')
plt.plot(test_label[:500],label='Home Office')
plt.title('XGBOOST FORECASTING')
plt.ylabel('HOME OFFICE ENERGY: Real VS Predicted')
plt.xlabel('Minutes')
plt.legend()
plt.show()


'''Cross Validation Score'''
#from sklearn.model_selection import cross_val_score
#CVSCORES=cross_val_score(model, x_test1, test_label, cv=6)
#CVSc08=(0.97400428+0.97187908+0.97246983+0.97336586+0.97399904+ 0.97331387)/6
#CVSc033=(0.95355326+ 0.95696911+ 0.94987157+ 0.95428189+ 0.95530962+0.9570078 )/6    
 
"""Tree Graphviz"""
#
#import matplotlib.pyplot as plt
#import os
#os.getcwd()
#os.chdir('C:/Program Files (x86)/Graphviz2.38/bin')
#xgb.plot_tree(model,num_trees=9)
#plt.rcParams['figure.figsize'] = [500, 100]
#plt.show()