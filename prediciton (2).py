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
from haversine import haversine
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
data = pd.read_csv("../input/nyc-taxi-trip-duration/train.zip")
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
if 'passenger_count' not in data.columns:
    plt.figure(figsize = (20,5))
    sns.boxplot(data.passenger_count)
    plt.show()
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
X = data.iloc[:,range(5,42)].values

X1 = np.append(arr = np.ones((X.shape[0],1)).astype(int), values = X, axis = 1)
X1.shape

import statsmodels.api as sm
X_opt = X1[:,range(0,37)]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()

#Fetch p values for each feature
p_Vals = regressor_OLS.pvalues

#define significance level for accepting the feature.
sig_Level = 0.05

#Loop to iterate over features and remove the feature with p value less than the sig_level
while max(p_Vals) > sig_Level:
    print("Probability values of each feature \n")
    print(p_Vals)
    X_opt = np.delete(X_opt, np.argmax(p_Vals), axis = 1)
    print("\n")
    print("Feature at index {} is removed \n".format(str(np.argmax(p_Vals))))
    print(str(X_opt.shape[1]-1) + " dimensions remaining now... \n")
    regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
    p_Vals = regressor_OLS.pvalues
    print("=================================================================\n")
    
#Print final summary
print("Final stat summary with optimal {} features".format(str(X_opt.shape[1]-1)))
regressor_OLS.summary()

#Split raw data
X_train, X_test, y_train, y_test = train_test_split(X,Y, random_state=4, test_size=0.2)

#Split data from the feature selection group
X_train_fs, X_test_fs, y_train_fs, y_test_fs = train_test_split(X_opt,Y, random_state=4, test_size=0.2)

#Split data from the feature extraction group
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X,Y, random_state=4, test_size=0.2)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_pca = scaler.fit_transform(X_train_pca)
X_test_pca = scaler.transform(X_test_pca)

from sklearn.decomposition import PCA
pca_10 = PCA(n_components=32)
X_train_pca = pca_10.fit_transform(X_train_pca)
X_test_pca = pca_10.transform(X_test_pca)

#Random Forest Regression
#instantiate the object for the Random Forest Regressor with default params from raw data
regressor_rfraw = RandomForestRegressor(n_jobs=-1)

#instantiate the object for the Random Forest Regressor with default params for Feature Selection Group
regressor_rf = RandomForestRegressor(n_jobs=-1)

#instantiate the object for the Random Forest Regressor for Feature Extraction Group
regressor_rf2 = RandomForestRegressor(n_jobs=-1)


#Train the object with default params for raw data
regressor_rfraw.fit(X_train,y_train)

#Train the object with default params for Feature Selection Group
regressor_rf.fit(X_train_fs,y_train_fs)

# #Train the object with default params for Feature Extraction Group
regressor_rf2.fit(X_train_pca,y_train_pca)

#Predict the output with object of default params for Feature Selection Group
y_pred_rfraw = regressor_rfraw.predict(X_test)

#Predict the output with object of default params for Feature Selection Group
y_pred_rf = regressor_rf.predict(X_test_fs)

#Predict the output with object of PCA params for Feature Extraction Group
y_pred_rfpca = regressor_rf2.predict(X_test_pca)

#Evaluate the model with default params for raw data
print('RMSE score for the RF regressor raw is : {}'.format(np.sqrt(metrics.mean_squared_error(y_test,y_pred_rfraw))))
print('RMSLE score for the RF regressor raw is : {}'.format(np.sqrt(metrics.mean_squared_log_error(y_test,y_pred_rfraw))))
print('Variance score for the RF regressor raw is : %.2f' % regressor_rfraw.score(X_test, y_test))

#Evaluate the model with default params for Feature Selection Group
print('RMSE score for the RF regressor is : {}'.format(np.sqrt(metrics.mean_squared_error(y_test_fs,y_pred_rf))))
print('RMSLE score for the RF regressor is : {}'.format(np.sqrt(metrics.mean_squared_log_error(y_test_fs,y_pred_rf))))
print('Variance score for the RF regressor is : %.2f' % regressor_rf.score(X_test_fs, y_test_fs))

#Evaluate the model with PCA params  for Feature Extraction Group
print('RMSE score for the RF regressor2 is : {}'.format(np.sqrt(metrics.mean_squared_error(y_test_pca, y_pred_rfpca))))
print('Variance score for the RF regressor2 is : %.2f' % regressor_rf2.score(X_test_pca, y_test_pca))

#exporting model
import pickle
import joblib
filename = 'random.pkl'
joblib.dump(regressor_rf2, filename)

#get accuracy
from sklearn.metrics import r2_score 
r2_score(y_test, y_pred_pca)

#XGBoost Regression

regressor_xgb = XGBRegressor(n_estimators=300,
                            learning_rate=0.5,
                            gamma=0,
                            subsample=0.75,
                            colsample_bytree=1,
                            max_depth=7,
                            min_child_weight=4,
                           n_jobs=-1)

#Train the object with default params for Feature Extraction Group
regressor_xgb.fit(X_train_pca,y_train_pca)

y_pred_xgb_pca = regressor_xgb.predict(X_test_pca)

#Evaluate the model with PCA params  for Feature Extraction Group
print('RMSE score for the XGBoost regressor2 is : {}'.format(np.sqrt(metrics.mean_squared_error(y_test_pca, y_pred_xgb_pca))))
print('Variance score for the XGBoost regressor2 is : %.2f' % regressor_xgb.score(X_test_pca, y_test_pca))

import pickle
import joblib
filename = 'xgboost.pkl'
joblib.dump(regressor_xgb, filename)

#get accuracy
r2_score(y_test, y_pred_pca)