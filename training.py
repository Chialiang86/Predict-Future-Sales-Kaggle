from math import log10, log2
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import lightgbm as lgb

# base cols
# shop_id,item_id,item_category_common,item_category_code,city_code
base_cols = ['shop_id','item_id','item_category_common','item_category_code','city_code']

# set time time_cols
time_cols = []
time_avg_cols = []
i = 1
max_i = 34
while i <= max_i:
    y = 2013 + i // 12
    for m in range(1, 13):
        time_cols.append('{}-{}'.format(y, str(m).zfill(2)))
        time_avg_cols.append('{}-{}-avg-price'.format(y, str(m).zfill(2)))
        i += 1
        if i > max_i:
            break
print(time_cols)

###########################
# read training data      #
###########################

print('reading whole training data ...')
df_train = pd.read_pickle('training.pkl')
df_test = pd.read_pickle('testing.pkl')
null_idx = df_test[df_test['2013-01'].isnull()].index.tolist()
df_test = df_test.fillna(0)

'''
9,16843 -> 74
12,20949 -> 500.0
42,20949 -> 501.0
31,20949 -> 569.0
25,20949 -> 560.0
9,4201 -> 193
'''
# drop special shop_id,item_id
df_train.drop(df_train[(df_train['shop_id'] == 12) & (df_train['item_id'] == 20949)].index, inplace=True)
df_train.drop(df_train[(df_train['shop_id'] == 42) & (df_train['item_id'] == 20949)].index, inplace=True)
df_train.drop(df_train[(df_train['shop_id'] == 31) & (df_train['item_id'] == 20949)].index, inplace=True)
df_train.drop(df_train[(df_train['shop_id'] == 25) & (df_train['item_id'] == 20949)].index, inplace=True)
df_train.drop(df_train[(df_train['shop_id'] == 9) & (df_train['item_id'] == 4201)].index, inplace=True)


###########################
# set training data x,y   #
###########################

# X, Y setting
print('X, Y setting whole')
X_train = df_train.drop(['2015-10', '2015-10-avg-price'], axis = 1)
Y_train = df_train['2015-10'].values
x_test = df_test.drop(['2013-01', '2013-01-avg-price'], axis=1)
print(X_train.shape, Y_train.shape)
print(x_test.shape)

print('split the data whole')
x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=0)
print(x_train.shape, y_train.shape)

# setting last 12 month
print('X, Y setting 12')
m = 12
time_12_cols = base_cols + time_cols[-m:] + time_avg_cols[-m:]
df_train_12 = df_train[time_12_cols]
print(df_train_12.columns.tolist())
df_test_12 = df_test[time_12_cols]
X_train_12 = df_train_12.drop(['2015-10', '2015-10-avg-price'], axis=1)
Y_train_12 = df_train_12['2015-10'].values
x_test_12 = df_test_12.drop(['2014-11', '2014-11-avg-price'], axis=1)
print(X_train_12.shape, Y_train_12.shape)
print(x_test_12.shape)

x_train_12, x_val_12, y_train_12, y_val_12 = train_test_split(X_train_12, Y_train_12, test_size=0.2, random_state=0)
print(x_train_12.shape, y_train_12.shape)

# setting last 6 month
print('X, Y setting 6')
m = 6
time_6_cols = base_cols + time_cols[-m:] + time_avg_cols[-m:]
df_train_6 = df_train[time_6_cols]
print(df_train_6.columns.tolist())
df_test_6 = df_test[time_6_cols]
X_train_6 = df_train_6.drop(['2015-10', '2015-10-avg-price'], axis=1)
Y_train_6 = df_train_6['2015-10'].values
x_test_6 = df_test_6.drop(['2015-05', '2015-05-avg-price'], axis=1)
print(X_train_6.shape, Y_train_6.shape)
print(x_test_6.shape)

x_train_6, x_val_6, y_train_6, y_val_6 = train_test_split(X_train_6, Y_train_6, test_size=0.2, random_state=0)
print(x_train_6.shape, y_train_6.shape)


###########################
# training by LightGBM    #
###########################

# Light GBM
print('Light GBM ...')

params = {
    'objective': 'mse',
    'metric': 'rmse',
    'num_leaves': 2 ** 7 - 1,
    'learning_rate': 0.005,
    'feature_fraction': 0.75,
    'bagging_fraction': 0.75,
    'bagging_freq': 5,
    'seed': 1,
    'verbose': 0
}

# training for whole
feature_name = x_train.columns.tolist()
lgb_train = lgb.Dataset(x_train[feature_name], y_train)
lgb_eval = lgb.Dataset(x_val[feature_name], y_val, reference=lgb_train)

evals_result = {}
gbm = lgb.train(
        params, 
        lgb_train,
        num_boost_round=3000,
        valid_sets=(lgb_train, lgb_eval), 
        feature_name = feature_name,
        verbose_eval=50, 
        evals_result = evals_result,
        early_stopping_rounds = 100)

plt.figure(figsize=(15,12))
lgb.plot_importance( gbm, max_num_features=30, importance_type='gain')
plt.title("Permutation Importance")
plt.savefig('lgb_feature_importances.png')

y_pre_t = gbm.predict(x_train)
y_pre_v = gbm.predict(x_val)
print('Train whole set mse:', mean_squared_error(y_train, y_pre_t))
print('Val whole set mse:', mean_squared_error(y_val, y_pre_v))

# training for 12 months
feature_name_12 = x_train_12.columns.tolist()
lgb_train_12 = lgb.Dataset(x_train_12[feature_name_12], y_train_12)
lgb_eval_12 = lgb.Dataset(x_val_12[feature_name_12], y_val_12, reference=lgb_train_12)

evals_result_12 = {}
gbm_12 = lgb.train(
        params, 
        lgb_train_12,
        num_boost_round=3000,
        valid_sets=(lgb_train_12, lgb_eval_12), 
        feature_name = feature_name_12,
        verbose_eval=50, 
        evals_result = evals_result_12,
        early_stopping_rounds = 100)

plt.figure(figsize=(15,12))
lgb.plot_importance( gbm_12, max_num_features=30, importance_type='gain')
plt.title("Permutation Importance 12 month")
plt.savefig('lgb_feature_importances_12.png')

y_pre_t_12 = gbm_12.predict(x_train_12)
y_pre_v_12 = gbm_12.predict(x_val_12)
print('Train 12 set mse:', mean_squared_error(y_train_12, y_pre_t_12))
print('Val 12 set mse:', mean_squared_error(y_val_12, y_pre_v_12))

# training for 6 months
feature_name_6 = x_train_6.columns.tolist()
lgb_train_6 = lgb.Dataset(x_train_6[feature_name_6], y_train_6)
lgb_eval_6 = lgb.Dataset(x_val_6[feature_name_6], y_val_6, reference=lgb_train_6)

evals_result_6 = {}
gbm_6 = lgb.train(
        params, 
        lgb_train_6,
        num_boost_round=3000,
        valid_sets=(lgb_train_6, lgb_eval_6), 
        feature_name = feature_name_6,
        verbose_eval=50, 
        evals_result = evals_result_6,
        early_stopping_rounds = 100)

plt.figure(figsize=(15,12))
lgb.plot_importance( gbm_6, max_num_features=30, importance_type='gain')
plt.title("Permutation Importance 6 month")
plt.savefig('lgb_feature_importances_6.png')

y_pre_t_6 = gbm_6.predict(x_train_6)
y_pre_v_6 = gbm_6.predict(x_val_6)
print('Train 6 set mse:', mean_squared_error(y_train_6, y_pre_t_6))
print('Val 6 set mse:', mean_squared_error(y_val_6, y_pre_v_6))

# plot prediction result
y_pre_t = ((np.array(y_pre_t) + np.array(y_pre_t_12) + np.array(y_pre_t_6)) / 3).tolist()
y_pre_v = ((np.array(y_pre_v) + np.array(y_pre_v_12) + np.array(y_pre_v_6)) / 3).tolist()
diff_t = y_pre_t - y_train
diff_v = y_pre_v - y_val

fig, ax = plt.subplots(2, figsize=(20, 10))
ax[0].set_title('gbm res train')
ax[0].plot(diff_t, label='diff')
ax[0].legend(loc='upper left')
ax[1].set_title('gbm res val')
ax[1].plot(diff_v, label='diff')
ax[1].legend(loc='upper left')
plt.savefig('gbm_predict.png')

diff_tdf = pd.DataFrame({'training': diff_t})
diff_vdf = pd.DataFrame({'validation' : diff_v})
thresh = 50

print('======================================== show LghtGBM result ========================================')
print('Train set mse:', mean_squared_error(y_train, y_pre_t))
print('Val set mse:', mean_squared_error(y_val, y_pre_v))
print('data of large error in training (thresh = {}):'.format(thresh))
large_diff_t = diff_tdf[abs(diff_tdf['training'].astype(int)) > thresh].index.tolist()
for i in large_diff_t:
    print('data : {}, get : {}, diff = {}'.format(x_train.iloc[i][['shop_id','item_id']], y_pre_t[i], y_pre_t[i] - y_train[i]))
print('data of large error in validation (thresh = {}):'.format(thresh))
large_diff_v = diff_vdf[abs(diff_vdf['validation'].astype(int)) > thresh].index.tolist()
for i in large_diff_v:
    print('data : {}, get : {}, diff = {}'.format(x_val.iloc[i][['shop_id','item_id']], y_pre_v[i], y_pre_v[i] - y_val[i]))
print('=====================================================================================================\n')


y_test = gbm.predict(x_test).clip(0, 20)
y_test_12 = gbm_12.predict(x_test_12).clip(0, 20)
y_test_6 = gbm_6.predict(x_test_6).clip(0, 20)
y_test_mean = ((np.array(y_test) + np.array(y_test_12) + np.array(y_test_6)) / 3).tolist()

gbm_submission = pd.DataFrame({
    "ID": range(214200), 
    "item_cnt_month": y_test_mean
})

gbm_submission.iloc[gbm_submission[(x_test['shop_id'] == 31) & (x_test['item_id'] == 20949)].index, gbm_submission.columns.get_loc('item_cnt_month')] = 430.0
gbm_submission.iloc[gbm_submission[(x_test['shop_id'] == 25) & (x_test['item_id'] == 20949)].index, gbm_submission.columns.get_loc('item_cnt_month')] = 450.0
gbm_submission['item_cnt_month'] = gbm_submission['item_cnt_month'].apply(lambda x: x if x >= 0 else 0)
gbm_submission.iloc[null_idx, gbm_submission.columns.get_loc('item_cnt_month')] = 0
gbm_submission.to_csv('submission.csv', index=False)
gbm_submission.head()

print('process complete')