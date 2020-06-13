import numpy as np
import pandas as pd

dataset=pd.read_csv('C:/Users/HP/Desktop/SmartHomeData.csv', delimiter=';')
Use=dataset['use [kW]']
Use=np.array(Use).reshape(-1,1)

Data=dataset.drop('use [kW]',axis=1)
L=['gen [kW]','House overall [kW]']
for i in L:
    Data=Data.drop(i,axis=1)


from sklearn.decomposition import PCA  

for j in range (17,12,-1):
    
    pca=PCA(n_components=j)
    Decomp=pca.fit(Data)
    data= pca.transform(Data)
  
    
    from sklearn.model_selection import train_test_split
    x_train1,x_test1,train_label,test_label=train_test_split(data, Use, test_size=0.33,random_state=0)


    """Multiple Linear Regression"""
    from sklearn.linear_model import LinearRegression

    import time
    depart=time.time()
    mod = LinearRegression()
    mod.fit(x_train1, train_label)

    temps=time.time()-depart
#
#
#      ## a Value
#    print("a Values" ,mod.coef_[0])
#
#
#        ## b Value
#    print("b Value" ,mod.intercept_[0])
#
#
#      ##RMSE 
#    from sklearn.metrics import mean_squared_error
#    RMSE=np.sqrt(mean_squared_error(test_label, mod.predict(x_test1)))
#    print("RMSE",RMSE)


    from sklearn.metrics import explained_variance_score
    EVS=explained_variance_score(mod.predict(x_test1), test_label)
    print("EV" ,EVS)


      ##R Squarred
    print("New score {r}: ".format(r=j) +str(100 * mod.score(x_test1,test_label)))
    print("****************************************************************************")
    
    
    