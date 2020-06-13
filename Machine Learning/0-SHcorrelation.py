import pandas as pd
import numpy as np 

##Data import
dataset=pd.read_csv('C:/Users/HP/Desktop/SmartHomeData.csv', delimiter=';')

Fur=np.array(dataset['Furnace 1 [kW]']).reshape(-1,1)
Tmp=np.array(dataset['temperature']).reshape(-1,1)
Data=dataset.drop('House overall [kW]',axis=1)
L=['gen [kW]','apparentTemperature']
for i in L:
    Data=Data.drop(i,axis=1)


import matplotlib.pyplot as plt
# Correlation matrix Define
def CorrelationMatrix(df, graphWidth):
    # drop columns with NaN
    df = df.dropna('columns') 
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=100, facecolor='w' )
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title('Correlation Matrix', fontsize=15)
    plt.show()


CorrelationMatrix(Data, 8)
CorrelationMatrix(dataset, 8)




#
#'''Confusion Matrix'''
#import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
#class_names=[0,1] # name  of classes
#fig, ax = plt.subplots()
#tick_marks = np.arange(len(class_names))
#plt.xticks(tick_marks, class_names)
#plt.yticks(tick_marks, class_names)
## create heatmap
#sns.heatmap(pd.DataFrame(CM), annot=True, cmap="YlGnBu" ,fmt='g')
#ax.xaxis.set_label_position("top")
#plt.tight_layout()
#plt.title('Confusion matrix', y=1.1)
#plt.ylabel('Actual label')
#plt.xlabel('Predicted label')￼