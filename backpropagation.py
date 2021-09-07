from sklearn.neural_network import MLPRegressor
import pandas as pd 
import numpy as np

datos = pd.read_csv('dataset.csv')

TempMax=datos['Max Temperature'].values
TempMin=datos['Min Temperature'].values
Precip=datos['Precipitation'].values
Hum=datos['Relative Humidity'].values
Solar=datos['Solar'].values
rendimiento = datos['rendimiento_kgxha'].values

X = np.array([TempMax,TempMin,Precip,Hum,Solar]).T
y = np.array(rendimiento)
clf = MLPRegressor(solver='lbfgs',
        hidden_layer_sizes=(100,
        100,
        100,
        100,
        100,
        100,
        100),
        random_state=1,
        max_iter=20000).fit(X,y)


# Datos para prediccion 1
# TempMax=27.1811065573771
# TempMin=12.2936475409836
# Precip=0.725098888977049
# Hum=0.52025912949006
# Solar=25.0963890668852

# Datos para prediccion 2
TempMax=27.6997459016393
TempMin=12.365868852459
Precip=0.355184542563934
Hum=0.490368366377355
Solar=25.7475268804796

# Datos para prediccion 3
# TempMax=24.3407213114754
# TempMin=12.5519508196721
# Precip=2.23538317241803
# Hum=0.722326809170853
# Solar=24.4206206425727

# Datos para prediccion 4
# TempMax=28.5302295081967
# TempMin=13.5499098360656
# Precip=0.92117227057541
# Hum=0.49846206861865
# Solar=21.1198588117646

# Datos para prediccion 5
# TempMax=23.9370849056604
# TempMin=11.4557688679245
# Precip=0.590763705138679
# Hum=0.62707443936853
# Solar=17.0758872239523

score = clf.score(X,y)
print(score)

result = clf.predict([[TempMax, TempMin, Precip, Hum, Solar]])
print(result)


# Datos de prediccion                                                                                    Resultados
# 9/1/2015,27.1811065573771,12.2936475409836,0.725098888977049,0.52025912949006,25.0963890668852,4081    3451
# 9/1/2016,27.6997459016393,12.365868852459,0.355184542563934,0.490368366377355,25.7475268804796,4020    3751
# 9/1/2017,24.3407213114754,12.5519508196721,2.23538317241803,0.722326809170853,24.4206206425727,3063    2671
# 9/1/2018,28.5302295081967,13.5499098360656,0.92117227057541,0.49846206861865,21.1198588117646,4297     3034
# 9/1/2019,23.9370849056604,11.4557688679245,0.590763705138679,0.62707443936853,17.0758872239523,3756    2536

