import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

# read data
print('read data')
data_train = pd.read_csv("competitive-data-science-predict-future-sales/sales_train.csv")
data_test = pd.read_csv("competitive-data-science-predict-future-sales/test.csv")
item_categories = pd.read_csv("competitive-data-science-predict-future-sales/item_categories.csv")
item = pd.read_csv("competitive-data-science-predict-future-sales/item_categories.csv")
shops = pd.read_csv("competitive-data-science-predict-future-sales/shops.csv")
sample_submission = pd.read_csv("competitive-data-science-predict-future-sales/sample_submission.csv")

print(item_categories.head())

# Remove the missing values
print('Remove the missing values')
def missing_value(data):
    missing_data = pd.DataFrame({
        'Missing_count':data.isnull().sum(),
        'Missing_part':data.isnull().sum()/len(data)
    })
    missing_data = missing_data[missing_data['Missing_count']!=0]

    return missing_data

print(missing_value(data_train))

if (data_train.duplicated().sum() != 0 ):
    data_train.drop_duplicates(keep = 'first', inplace=True)
    print(data_train.duplicated().sum())

# change date type to datetime
print('change date type to datetime')
data_train['date'] = pd.to_datetime(data_train['date'])
data_train['date'] = data_train['date'].apply(lambda x: x.strftime('%Y-%m'))
print(data_train.head())

# plot pair
print('plot pair and scatter')
sns.pairplot(data_train)

plt.scatter(data_train['date'], data_train['item_price'])
plt.savefig('date-item_price.png')

# Remove the outliers (item_price)
print('Remove the outliers (item_price)')
print(data_train[data_train.item_price > 100000])
# data_train.drop(1163158,inplace = True)

# Drop columns
print('Drop column')
data_train.drop(['date_block_num', 'item_price'], axis=1, inplace=True)
print(data_train.head())

# Data train
print('Data train')
df_train = data_train.groupby(['date','shop_id','item_id']).sum()
df_train = df_train.pivot_table(index=['shop_id','item_id'], 
                                columns='date', 
                                values='item_cnt_day', 
                                fill_value=0)

df_train.reset_index(inplace=True)
df_train.head()

# Data test
print('Data test')
df_test = pd.merge(data_test, df_train, on=['shop_id','item_id'], how='left')
df_test.drop(['ID'], axis=1, inplace=True)
df_test = df_test.fillna(0)
df_test.head()

# X, Y setting
print('X, Y setting')
X_train = df_train.drop(['2015-12'], axis = 1)
Y_train = df_train['2015-12'].values

x_test = df_test.drop(['2013-01'], axis=1)
print(X_train.shape, Y_train.shape)
print(x_test.shape)

# Split the data
print('split the data')
x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=0)
print(X_train.shape)
print(y_train.shape)

# linear regression
print('linear regression')
LR = LinearRegression()
LR.fit(x_train, y_train)

print('Train set mse:', mean_squared_error(y_train, LR.predict(x_train)))
print('Val set mse:', mean_squared_error(y_val, LR.predict(x_val)))

# Gradient boosting Regression
print('boosting Regression')
gbr = GradientBoostingRegressor(learning_rate=0.1, n_estimators=100, max_depth=3)
gbr.fit(x_train, y_train)

print('Train set mse:', mean_squared_error(y_train, gbr.predict(x_train)))
print('Val set mse:', mean_squared_error(y_val, gbr.predict(x_val)))

# Predict using gradient boosting regression
pred_test = gbr.predict(x_test)

submission = pd.DataFrame({
    'ID':data_test['ID'],
    'item_cnt_month':pred_test
})
submission.to_csv('submission.csv', index=False)

submission.head()

print('process complete')