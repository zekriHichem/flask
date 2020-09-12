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
    sns.boxplot(data.passenger_count)
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

X = data.iloc[:,range(0,7)].values

#Split raw data
X_train, X_test, y_train, y_test = train_test_split(X,Y, random_state=4, test_size=0.2)

regressor_xgb = XGBRegressor(n_estimators=300,
                            learning_rate=0.08,
                            gamma=0,
                            subsample=0.75,
                            colsample_bytree=1,
                            max_depth=7,
                            min_child_weight=4,
                           n_jobs=-1)

#Train the object with default params for Feature Extraction Group
regressor_xgb.fit(X_train,y_train)

y_pred_xgb = regressor_xgb.predict(X_test)

#Evaluate the model with PCA params  for Feature Extraction Group
print('RMSE score for the XGBoost regressor2 is : {}'.format(np.sqrt(metrics.mean_squared_error(y_test, y_pred_xgb))))
print('Variance score for the XGBoost regressor2 is : %.2f' % regressor_xgb.score(X_test, y_test))
r2_score(y_test, y_pred)