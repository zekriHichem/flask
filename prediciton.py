
import numpy as np
import pandas as pd
import seaborn as sns
import json as js
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
import statsmodels.formula.api as sm
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import warnings; warnings.simplefilter('ignore')

#parameters
x =  '{ "lat_min":40.5, "lat_max":40.9, "lon_min":-74.2, "lon_max":-73.7}'
params = js.loads(x)

# Function for calculating distance
def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

#importing dataset
data = pd.read_csv("train.zip")
if 'distance' not in data.columns:
    data['distance'] = haversine_np(data['pickup_longitude'], 
    data['pickup_latitude'], 
    data['dropoff_longitude'], 
    data['dropoff_latitude'])
data.rename(columns={'trip_duration':'duration',
                        'pickup_latitude':'pickupLat',
                        'pickup_longitude':'pickupLng',
                        'dropoff_latitude':'destinationLat',
                        'dropoff_longitude':'destinationLng'}, 
                 inplace=True)

#passenger count analysis
if 'passenger_count' in data.columns:
    plt.figure(figsize = (20,5))
    data.passenger_count.describe()
    # turning 0 passengers to 1
    data['passenger_count'] = data.passenger_count.map(lambda x: 1 if x == 0 else x)
    data = data[data.passenger_count <= 6]
    data.passenger_count.value_counts()

if 'vendor_id' in data.columns:
    del data['vendor_id']
if 'dropoff_datetime' in data.columns:
    del data['dropoff_datetime']
if 'passenger_count' in data.columns:
    del data['passenger_count']
if 'store_and_fwd_flag' in data.columns:
    del data['store_and_fwd_flag']

#Converting Timestamp to datetime
data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])

data['month'] = data.pickup_datetime.dt.month
data['weekday_num'] = data.pickup_datetime.dt.weekday
data['pickup_hour'] = data.pickup_datetime.dt.hour

dummy = pd.get_dummies(data.month, prefix='month')
dummy.drop(dummy.columns[0], axis=1, inplace=True) #avoid dummy trap
data = pd.concat([data,dummy], axis = 1)

dummy = pd.get_dummies(data.weekday_num, prefix='weekday_num')
dummy.drop(dummy.columns[0], axis=1, inplace=True) #avoid dummy trap
data = pd.concat([data,dummy], axis = 1)

dummy = pd.get_dummies(data.pickup_hour, prefix='pickup_hour')
dummy.drop(dummy.columns[0], axis=1, inplace=True) #avoid dummy trap
data = pd.concat([data,dummy], axis = 1)

#removing data with duration more than 24h
data = data[data.duration <= 86400]

#removing data of passengers who took more than 1 minute to cancel a trip
data = data[~((data.distance == 0) & (data.duration >= 60))]

#remove data of passengers who took more than 1 hour in short distance trips
data = data[~((data['distance'] <= 1) & (data['duration'] >= 3600))]

data = data[data.pickupLng != data.pickupLng.min()]

del data['id']
del data['pickup_datetime']

Y = data.iloc[:,4].values
del data['duration']
X = data.values
import umap
reducer = umap.UMAP()
embedding = reducer.fit_transform(X)
print(embedding.shape)
from sklearn.linear_model import LinearRegression
reg = LinearRegression(normalize = True , copy_X = False,fit_intercept = False, n_jobs = -1 ).fit(embedding, Y)
reg.score(embedding, Y) * 100
