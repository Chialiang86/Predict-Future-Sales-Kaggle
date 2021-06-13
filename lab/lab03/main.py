from math import log10, log2
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

# read data
print('reading data...')
data_train = pd.read_csv("competitive-data-science-predict-future-sales/sales_train.csv")
data_test = pd.read_csv("competitive-data-science-predict-future-sales/test.csv")
item_categories = pd.read_csv("competitive-data-science-predict-future-sales/item_categories.csv")
items = pd.read_csv("competitive-data-science-predict-future-sales/items.csv")
shops = pd.read_csv("competitive-data-science-predict-future-sales/shops.csv")
sample_submission = pd.read_csv("competitive-data-science-predict-future-sales/sample_submission.csv")
print(item_categories.head())

# set item_id -> item_category_id dict
print('set item_id -> item_category_id dict...')
item_to_cate = {}
for i in range(len(items)):
    item_to_cate[items['item_id'].iloc[i]] = items['item_category_id'].iloc[i]

# Remove duplicate value
if (data_train.duplicated().sum() != 0 ):
    data_train.drop_duplicates(keep = 'first', inplace=True)
    print(data_train.duplicated().sum())

# Remove the outliers (item_price)
print('Remove the outliers (item_price)...')
data_train.drop(data_train[data_train['item_price'] > 100000].index, inplace=True)
print('Remove the outliers (item_cnt_day)...')
data_train.drop(data_train[data_train['item_cnt_day'] > 1500].index, inplace=True)

# change date type to datetime
print('change date type to datetime...')
data_train['date'].astype(str)
data_train['date'] = data_train['date'].apply(lambda x: (x.split('.')[2] + '-' + x.split('.')[1]))
print(data_train)

# Drop columns
print('Drop column')
data_train.drop(['date_block_num', 'item_price'], axis=1, inplace=True)

# Data train
print('Data train preprocessing...')
print('groupby setting ...')
df_train = data_train.groupby(['date','shop_id','item_id']).sum() 
print('pivot_table setting ...')
df_train = df_train.pivot_table(index=['shop_id','item_id'], 
                                columns='date', 
                                values='item_cnt_day', 
                                fill_value=0)
# df_train['item_price_avg'] = data_train.groupby(['shop_id','item_id'])['item_price'].mean()
# df_train['item_price_avg'] = df_train['item_price_avg'].apply(lambda x: log10(x)) # setting weight

# reset index
print('reset_index ...')
df_train.reset_index(inplace=True)
print(df_train.head())

# rest order of col
print('rest order of col')
cols = ['shop_id', 'item_id']
i = 1
while i < 35:
    y = 2013 + i // 12
    for m in range(1, 13):
        cols.append('{}-{}'.format(y, str(m).zfill(2)))
        i += 1
        if i >= 35:
            break
df_train = pd.DataFrame(df_train)[cols]
print(df_train)

# Data test
print('Data test')
df_test = pd.merge(data_test, df_train, on=['shop_id','item_id'], how='left')
df_test.drop(['ID'], axis=1, inplace=True)
df_test = df_test.fillna(0)
print(data_test.head())

# update df_train
print('replacing item id of df_train by cate id...')
df_train['item_id'] = df_train['item_id'].apply(lambda x: item_to_cate[x])
print(df_train.head())

# update df_test
print('replacing item id of df_test by cate id...')
df_test['item_id'] = df_test['item_id'].apply(lambda x: item_to_cate[x])
print(df_test.head())

# X, Y setting
print('X, Y setting')
X_train = df_train.drop(['2015-10'], axis = 1)
Y_train = df_train['2015-10'].values

print('test setting')
x_test = df_test.drop(['2013-01'], axis=1)

# Split the data
print('split the data')
x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=0)
print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)

# linear regression
print('linear regression')
LR = LinearRegression()
LR.fit(x_train, y_train)

# plot prediction result
y_pre_t = LR.predict(x_train)
y_pre_v = LR.predict(x_val)
diff_t = y_pre_t - y_train
diff_v = y_pre_v - y_val


fig, ax = plt.subplots(2, figsize=(20, 10))
ax[0].set_title('LR res train')
ax[0].plot(diff_t, label='diff')
ax[0].legend(loc='upper left')
ax[1].set_title('LR res val')
ax[1].plot(diff_v, label='diff')
ax[1].legend(loc='upper left')
plt.savefig('LR_predict.png')

diff_tdf = pd.DataFrame({'training': diff_t})
diff_vdf = pd.DataFrame({'validation' : diff_v})
thresh = 50

print('============= show LR result =============')
print('Train set mse:', mean_squared_error(y_train, y_pre_t))
print('Val set mse:', mean_squared_error(y_val, y_pre_v))
print('data of large error in training (thresh = {}):'.format(thresh))
print(x_train.iloc[diff_tdf[abs(diff_tdf['training'].astype(int)) > thresh].index.tolist()])
print('data of large error in validation (thresh = {}):'.format(thresh))
print(x_val.iloc[diff_vdf[abs(diff_vdf['validation'].astype(int)) > thresh].index.tolist()])
print('==========================================\n')

# Gradient boosting Regression
print('boosting Regression')
gbr = GradientBoostingRegressor(learning_rate=0.1, n_estimators=100, max_depth=3)
gbr.fit(x_train, y_train)

# plot prediction result
y_pre_t = gbr.predict(x_train)
y_pre_v = gbr.predict(x_val)
diff_t = y_pre_t - y_train
diff_v = y_pre_v - y_val

fig, ax = plt.subplots(2, figsize=(20, 10))
ax[0].set_title('gbr res train')
ax[0].plot(diff_t, label='diff')
ax[0].legend(loc='upper left')
ax[1].set_title('gbr res val')
ax[1].plot(diff_v, label='diff')
ax[1].legend(loc='upper left')
plt.savefig('gbr_predict.png')

diff_tdf = pd.DataFrame({'training': diff_t})
diff_vdf = pd.DataFrame({'validation' : diff_v})
thresh = 50

print('============= show gbr result =============')
print('Train set mse:', mean_squared_error(y_train, y_pre_t))
print('Val set mse:', mean_squared_error(y_val, y_pre_v))
print('data of large error in training (thresh = {}):'.format(thresh))
print(x_train.iloc[diff_tdf[abs(diff_tdf['training'].astype(int)) > thresh].index.tolist()])
print('data of large error in validation (thresh = {}):'.format(thresh))
print(x_val.iloc[diff_vdf[abs(diff_vdf['validation'].astype(int)) > thresh].index.tolist()])
print('===========================================\n')

# XGBoost
print('xgboost Regression')
xgbc = XGBRegressor(base_score=0.5, booster='gbtree', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=None, subsample=1, verbosity=1)
xgbc.fit(x_train, y_train)


# plot prediction result
y_pre_t = xgbc.predict(x_train)
y_pre_v = xgbc.predict(x_val)
diff_t = y_pre_t - y_train
diff_v = y_pre_v - y_val

fig, ax = plt.subplots(2, figsize=(20, 10))
ax[0].set_title('xgbc res train')
ax[0].plot(diff_t, label='diff')
ax[0].legend(loc='upper left')
ax[1].set_title('xgbc res val')
ax[1].plot(diff_v, label='diff')
ax[1].legend(loc='upper left')
plt.savefig('xgbc_predict.png')

diff_tdf = pd.DataFrame({'training': diff_t})
diff_vdf = pd.DataFrame({'validation' : diff_v})
thresh = 50

print('============= show xgbc result =============')
print('Train set mse:', mean_squared_error(y_train, y_pre_t))
print('Val set mse:', mean_squared_error(y_val, y_pre_v))
print('data of large error in training (thresh = {}):'.format(thresh))
print(x_train.iloc[diff_tdf[abs(diff_tdf['training'].astype(int)) > thresh].index.tolist()])
print('data of large error in validation (thresh = {}):'.format(thresh))
print(x_val.iloc[diff_vdf[abs(diff_vdf['validation'].astype(int)) > thresh].index.tolist()])
print('============================================\n')

# Predict using gradient boosting regression
pred_test = gbr.predict(x_test)



submission = pd.DataFrame({
    'ID':data_test['ID'],
    'item_cnt_month':pred_test
})
submission.to_csv('submission.csv', index=False)

submission.head()

print('process complete')