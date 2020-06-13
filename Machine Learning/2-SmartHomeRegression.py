
import pandas as pd

##Data import
dataset=pd.read_csv('C:/Users/HP/Desktop/SmartHomeData.csv', delimiter=';')
Use=dataset['use [kW]']
Temperature=dataset['temperature']
Microwave=dataset['Microwave [kW]']
Fridge=dataset['Fridge [kW]']
Living=dataset['Living room [kW]']
Time=dataset['time']
Furnace2=dataset['Furnace 2 [kW]']

"""Reshape Arrays"""
import numpy as np
Use=np.array(Use).reshape(-1,1)
Temperature=np.array(Temperature).reshape(-1,1)
Microwave=np.array(Microwave).reshape(-1,1)
Fridge=np.array(Fridge).reshape(-1,1)
Living=np.array(Living).reshape(-1,1)
Time=np.array(Time).reshape(-1,1)
Furnace2=np.array(Furnace2).reshape(-1,1)

"""MAPE Define"""
from sklearn.utils import check_arrays
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = check_arrays(y_true, y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

"""Simple Linear Regression"""
from sklearn.linear_model import LinearRegression
L=[Temperature,Time,Furnace2]
model = LinearRegression()
for i in L:
    model.fit(i, Use)




        ## a Value
    print("a Value" ,model.coef_[0])


        ## b Value
    print("b Value" ,model.intercept_[0])

      ## Plot
    import matplotlib.pyplot as plt
    plt.plot(i,model.predict(Temperature),color='blue')
    plt.scatter(i,Use,color='green')
    plt.title('Linear Regression')
    plt.ylabel('Used Energy')
#    plt.xlabel('Index')
    plt.show()


     ##RMSE 
    from sklearn.metrics import mean_squared_error
    RMSE=np.sqrt(mean_squared_error(Use,model.predict(i)))
    print("RMSE",RMSE)


    from sklearn.metrics import explained_variance_score
    EVS=explained_variance_score(Use,model.predict(i) )
    print("EV" ,EVS)


    print("MAPE",mean_absolute_percentage_error(model.predict(i),Use))
 
#        ##R Squarred
#    print("Rsq",model.score(i,Use))
    
"""--------------------------------------- Multiple Linear Regression -----------------------------------------------------------------------------------"""

Data=dataset.drop('use [kW]',axis=1)
L=['gen [kW]','House overall [kW]']
for i in L:
    Data=Data.drop(i,axis=1)

from sklearn.model_selection import train_test_split
x_train1,x_test1,train_label,test_label=train_test_split(Data, Use, test_size=0.33,random_state=0)


"""Multiple Linear Regression"""
from sklearn.linear_model import LinearRegression

import time
depart=time.time()
mod = LinearRegression()
mod.fit(x_train1, train_label)

temps=time.time()-depart


      ## a Value
print("a Values" ,mod.coef_[0])


        ## b Value
print("b Value" ,mod.intercept_[0])


  ##RMSE 
from sklearn.metrics import mean_squared_error
RMSE=np.sqrt(mean_squared_error(mod.predict(x_test1),test_label))
print("RMSE",RMSE)

##EV
from sklearn.metrics import explained_variance_score
EVS=explained_variance_score( test_label,mod.predict(x_test1))
print("EV" ,EVS)


#  ##R Squarred
mod.score(x_test1,test_label)
print("MAPE",mean_absolute_percentage_error(mod.predict(x_test1),test_label))


