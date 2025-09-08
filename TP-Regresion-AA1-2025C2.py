import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler   # u otros scalers
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LassoCV, RidgeCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('uber_fares.csv')
print(data.head())
### Columnas, ¿cuáles son variables numéricas y cuales variables categóricas?
print(data.columns)
#Key - probablemente sacar
#Datos faltantes:
#dropoff_latitude, dropoff_longitude - falta 1 dato
print(data.columns.values)
#Habría que revisar los valores negativos <- como mucho capear a 0
#Preguntar si los valores negativos son reembolsos, y como identificarlos
#Clave natural -> Fecha, ubicacion salida y llegada, cantidad de pasajeros
#Preguntar al profe
sns.boxplot(data[['fare_amount']])
sns.boxplot(data['pickup_latitude'])
sns.boxplot(data['dropoff_latitude'])
sns.boxplot(data['dropoff_longitude'])
#208 pasajeros en un viaje - valor atípicos <- no influye en el costo del viaje, capear probablemente.
sns.boxplot(data[data['passenger_count'] <= 200]['passenger_count'])
data['passenger_count'].describe()
X_train, X_test, y_train, y_test = train_test_split(data.drop(columns='fare_amount'), data['fare_amount'], test_size=0.2, random_state=42)
print(X_train.describe())