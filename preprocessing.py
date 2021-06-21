from math import log10, log2
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

# read data
print('reading data...')
data_train = pd.read_csv("competitive-data-science-predict-future-sales/sales_train.csv")
data_test = pd.read_csv("competitive-data-science-predict-future-sales/test.csv")
item_categories = pd.read_csv("competitive-data-science-predict-future-sales/item_categories.csv")
items = pd.read_csv("competitive-data-science-predict-future-sales/items.csv")
shops = pd.read_csv("competitive-data-science-predict-future-sales/shops.csv")
print(item_categories.head())

# data_train.loc[data_train.shop_id == 0, 'shop_id'] = 57
# data_test.loc[data_test.shop_id == 0, 'shop_id'] = 57

# data_train.loc[data_train.shop_id == 1, 'shop_id'] = 58
# data_test.loc[data_test.shop_id == 1, 'shop_id'] = 58

# data_train.loc[data_train.shop_id == 40, 'shop_id'] = 39
# data_test.loc[data_test.shop_id == 40, 'shop_id'] = 39

# Remove duplicate value
if (data_train.duplicated().sum() != 0 ):
    data_train.drop_duplicates(keep = 'first', inplace=True)
    print(data_train.duplicated().sum())

# Remove the outliers (item_price)
print('Remove the outliers (item_price)...')
data_train.drop(data_train[data_train['item_price'] > 100000].index, inplace=True)
print('Remove the outliers (item_cnt_day)...')
data_train.drop(data_train[data_train['item_cnt_day'] > 1500].index, inplace=True)
idx = data_train[data_train['item_cnt_day'] < 0].index
data_train.at[idx, 'item_cnt_day'] = 0

# change date type to datetime
print('change date type to datetime...')
data_train['date'].astype(str)
data_train['date'] = data_train['date'].apply(lambda x: (x.split('.')[2] + '-' + x.split('.')[1]))
print(data_train)

# Drop columns
print('Drop column')
# data_train.drop(['date_block_num'], axis=1, inplace=True)


######################
# feature engenering #
######################

# Data train
print('Data train preprocessing...')
print('groupby setting ...')
df_train = data_train.groupby(['date','shop_id','item_id']).sum() 
print('pivot_table setting ...')
df_train = df_train.pivot_table(index=['shop_id','item_id'], 
                                columns='date', 
                                values='item_cnt_day', 
                                fill_value=0)
df_train.reset_index(inplace=True)
print(df_train.head())

# Data test
print('Data test')
df_test = pd.merge(data_test, df_train, on=['shop_id','item_id'], how='left')
df_test.drop(['ID'], axis=1, inplace=True)
print(data_test.head())

# setting avg price
print('setting avg price')

# set time time_cols
time_cols = []
i = 1
max_i = 34
while i <= max_i:
    y = 2013 + i // 12
    for m in range(1, 13):
        time_cols.append('{}-{}'.format(y, str(m).zfill(2)))
        i += 1
        if i > max_i:
            break
print(time_cols)

df_train_avg = data_train.groupby(['date','shop_id','item_id']).mean()
df_train_avg = df_train_avg.pivot_table(index=['shop_id','item_id'], 
                                columns='date', 
                                values='item_price',
                                fill_value=0)
for col in time_cols:
    df_train_avg.rename({col:col + '-avg-price'}, axis=1, inplace=True)
    df_train_avg[col + '-avg-price'] = df_train_avg[col + '-avg-price'].apply(lambda x: log2(x) if x > 0 else 0)
df_train = pd.merge(df_train, df_train_avg, on=['shop_id','item_id'], how='left')
df_test = pd.merge(df_test, df_train_avg, on=['shop_id','item_id'], how='left')

# df_train_avg['avg_price'] = df_train_avg[time_cols].max(axis=1)
# df_train_avg.reset_index(inplace=True)
# df_train_avg = df_train_avg[['shop_id','item_id','avg_price']]
# df_train_avg['avg_log_price'] = df_train_avg['avg_price'].apply(lambda x: log2(x))
# df_train = pd.merge(df_train, df_train_avg, on=['shop_id','item_id'], how='left')
# df_test = pd.merge(df_test, df_train_avg, on=['shop_id','item_id'], how='left')
df_train = df_train.fillna(0)
print(df_train.head())
print(df_test.head())

# item_category
print('set item_id -> item_category_id dict...')
map_dict = {
            'Чистые носители (штучные)': 'Чистые носители',
            'Чистые носители (шпиль)' : 'Чистые носители',
            'PC ': 'Аксессуары',
            'Служебные': 'Служебные '
            }

items = pd.merge(items, item_categories, on='item_category_id')

items['item_category'] = items['item_category_name'].apply(lambda x: x.split('-')[0])
items['item_category'] = items['item_category'].apply(lambda x: map_dict[x] if x in map_dict.keys() else x)
items['item_category_common'] = LabelEncoder().fit_transform(items['item_category'])
items['item_category_code'] = LabelEncoder().fit_transform(items['item_category_name'])
items = items[['item_id', 'item_category_common', 'item_category_code']]

df_train = pd.merge(df_train, items, on=['item_id'], how='left')
df_test = pd.merge(df_test, items, on=['item_id'], how='left')
print(df_train.head())
print(df_test.head())

# setting city
print('setting city...')
shops['city'] = shops['shop_name'].apply(lambda x: x.split()[0].lower())
shops.loc[shops.city == '!якутск', 'city'] = 'якутск'
shops['city_code'] = LabelEncoder().fit_transform(shops['city'])
shops = shops[['shop_id', 'city_code']]
df_train = pd.merge(df_train, shops, on=['shop_id'], how='left')
df_test = pd.merge(df_test, shops, on=['shop_id'], how='left')
print(df_train.head())
print(df_test.head())

# Setting first day
# print('Setting first day')
# df_first = data_train.groupby(['shop_id','item_id'])['date_block_num'].min().reset_index()
# df_first.rename({'date_block_num': 'data_first_sold'}, axis=1, inplace=True)
# df_train = pd.merge(df_train, df_first, on=['shop_id','item_id'], how='left')
# df_test = pd.merge(df_test, df_first, on=['shop_id','item_id'], how='left')
# df_train['data_first_sold'] = df_train['data_first_sold'].fillna(0)
# df_test['data_first_sold'] = df_test['data_first_sold'].fillna(0)
# df_train['data_first_sold'] = df_train['data_first_sold'].apply(lambda x: 1.5 ** (x - 17))
# df_test['data_first_sold'] = df_test['data_first_sold'].apply(lambda x: 1.5 ** (x - 17))
# print(df_train.head())
# print(df_test.head())

df_train.to_csv('training.csv')
df_test.to_csv('testing.csv')
df_train.to_pickle('training.pkl')
df_test.to_pickle('testing.pkl')

print('process complete')