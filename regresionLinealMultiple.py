#Estimacion de la cosacha en base a la temMax, temMin, precipitacion, Hum, SolarUV

# cosecha = B0temMAX + B1temMin + B2precip + B3Hum + B4SolarUV

import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

sns.set_style('darkgrid')

datos = pd.read_csv('dataset.csv') #Reemplazar por el nombre del archivo

nuevo = datos[['Date','Max Temperature','Min Temperature', 'Precipitation','Relative Humidity', 'Solar', 'rendimiento_kgxha']]



# g = sns.pairplot(nuevo, hue='Date', diag_kind='hist')
# for ax in g.axes.flat:
#     plt.setp(ax.get_xticklabels(), rotation=45)

TempMax=datos['Max Temperature'].values
TempMin=datos['Min Temperature'].values
Precip=datos['Precipitation'].values
Hum=datos['Relative Humidity'].values
Solar=datos['Solar'].values
rendimiento = datos['rendimiento_kgxha'].values
X = np.array([TempMax,TempMin,Precip,Hum,Solar]).T
y = np.array(rendimiento)


reg = LinearRegression()

reg=reg.fit(X,y)

Y_pred=reg.predict(X)

error=np.sqrt(mean_squared_error(y,Y_pred))
r2=reg.score(X,y)
print('El error es: ', error)
print('El valor de R2 es: ', r2)
print('Coeficientes: \n', reg.coef_)
TempMax=27.62
TempMin=12.09
Precip=0.37
Hum=0.48
Solar=25.6185
print('Rendimiento de la prediccion: \n', reg.predict([[TempMax,TempMin,Precip,Hum,Solar]]))

