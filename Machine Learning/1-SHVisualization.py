
import pandas as pd
Data=pd.read_csv('C:/Users/HP/Desktop/SmartHomeData.csv', delimiter=';')

##Time Modify
time = pd.date_range('2019-01-01 05:00', periods=len(Data),  freq='min')  
time = pd.DatetimeIndex(time)
Data['time']=time

##Data Cleaning
Data=Data.drop('House overall [kW]',axis=1)
Data=Data.drop('gen [kW]',axis=1)


#RoomName=['Home office [kW]','Wine cellar [kW]','Kitchen 12 [kW]','Kitchen 14 [kW]','Kitchen 38 [kW]','Barn [kW]','Living room [kW]']
#DeviceName=['Dishwasher [kW]','Furnace 1 [kW]','Furnace 2 [kW]','Fridge [kW]','Garage door [kW]','Well [kW]','Microwave [kW]','Solar [kW]']
#import numpy as np
#Rooms=np.empty([503910,0])
#x=0
#for i in range(7):
#    Rooms=np.append(Rooms, np.array(Data[RoomName[i]]).reshape(-1,1), axis=1)
#    x=x+Rooms[1,i]
#
#Devices=np.empty([503910,0])
#y=0
#for i in range(8):
#    Devices=np.append(Devices, np.array(Data[DeviceName[i]]).reshape(-1,1), axis=1)
#    y=y+Devices[1,i]
#
# x+y


##Date & Hour Representation
Data['day']= Data['time'].dt.day
Data['month']= Data['time'].dt.month
Data['week']= Data['time'].dt.week
Data['hour']= Data['time'].dt.hour


##Visualize Define 
import matplotlib.pyplot as plt
def visualize(label, cols):
    fig,ax=plt.subplots(figsize=(30,14))
    colour= ['C{r}'.format(r=str(i)) for i in range(20)]
    for colour,col in zip(colour,cols):
            Data.groupby(label)[col].mean().plot(ax=ax,label=col,color=colour)
    plt.legend()
   
#Viz..    
visualize('hour',['Furnace 1 [kW]','Furnace 2 [kW]','Fridge [kW]','Garage door [kW]','Well [kW]','Microwave [kW]','Solar [kW]','Dishwasher [kW]'])
visualize('day',['Furnace 1 [kW]','Furnace 2 [kW]','Fridge [kW]','Garage door [kW]','Well [kW]','Microwave [kW]','Solar [kW]','Dishwasher [kW]'])
visualize('month',['Furnace 1 [kW]','Furnace 2 [kW]','Fridge [kW]','Garage door [kW]','Well [kW]','Microwave [kW]','Solar [kW]','Dishwasher [kW]'])

visualize('hour',['Home office [kW]','Wine cellar [kW]','Kitchen 12 [kW]','Kitchen 14 [kW]','Kitchen 38 [kW]','Barn [kW]','Living room [kW]'])

visualize('month',['use [kW]'])


##Seaborn Visualisation 
import seaborn as sns
sns.set_style('darkgrid')
plt.xlim([0,0.2])
plt.ylim([0,3])
sns.distplot(Data['Living room [kW]'])


