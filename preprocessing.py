from math import log10, log2
import numpy as np
import pandas as pd

from pandas.core.frame import DataFrame
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

mpl.rcParams['agg.path.chunksize'] = 10000

# read data
print('reading data...')
data_train = pd.read_csv("competitive-data-science-predict-future-sales/sales_train.csv")
data_test = pd.read_csv("competitive-data-science-predict-future-sales/test.csv")
item_categories = pd.read_csv("competitive-data-science-predict-future-sales/item_categories.csv")
items = pd.read_csv("competitive-data-science-predict-future-sales/items.csv")
shops = pd.read_csv("competitive-data-science-predict-future-sales/shops.csv")
sample_submission = pd.read_csv("competitive-data-science-predict-future-sales/sample_submission.csv")
print(item_categories.head())

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
idx = data_train[data_train['item_cnt_day'] < 0].index
data_train.at[idx, 'item_cnt_day'] = 0


# change date type to datetime
print('change date type to datetime...')
data_train['date'].astype(str)
data_train['date'] = data_train['date'].apply(lambda x: (x.split('.')[2] + '-' + x.split('.')[1]))
print(data_train)

# Drop columns
print('Drop column')
data_train.drop(['date_block_num'], axis=1, inplace=True)

# Data train time series
print('Data train preprocessing...')
df_train = data_train.groupby(['date','shop_id','item_id']).sum()
df_train = df_train.pivot_table(index=['shop_id','item_id'], 
                                columns='date', 
                                values='item_cnt_day', 
                                fill_value=0)

# reset index
print('reset index ...')
df_train.reset_index(inplace=True)

# add attribute

# city
shops['city'] = shops['shop_name'].apply(lambda x: x.split()[0].lower())
shops.loc[shops.city == '!якутск', 'city'] = 'якутск'
shops['city_code'] = LabelEncoder().fit_transform(shops['city'])
shops = shops[['shop_id', 'city_code']]
df_train = pd.merge(df_train, shops, on=['shop_id'], how='left')

# avg price
df_train_avg = data_train.groupby(['date','shop_id','item_id']).mean()
df_train_avg = df_train_avg.pivot_table(index=['shop_id','item_id'], 
                                columns='date', 
                                values='item_price',
                                fill_value=0)
df_train_avg['avg_price'] = df_train_avg[time_cols].max(axis=1)
df_train_avg.reset_index(inplace=True)
df_train_avg = df_train_avg[['shop_id','item_id','avg_price']]
df_train = pd.merge(df_train, df_train_avg, on=['shop_id','item_id'], how='left')
print(df_train.head())

# item_cat
df_train['item_category_id'] = df_train['item_id'].apply(lambda x: item_to_cate[x])

# fill 0 to None
df_train = df_train.fillna(0)

print(df_train.head())

# Data test
print('Data test')
df_test = pd.merge(data_test, df_train, on=['shop_id','item_id'], how='left')
df_test.drop(['ID'], axis=1, inplace=True)
df_test = df_test.fillna(0)
print(data_test.head())

df_train.to_csv('training.csv')
df_test.to_csv('testing.csv')

print('processing complete.')