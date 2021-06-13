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


# Remove the outliers (item_price)
print('Remove the outliers (item_price)')
data_train.drop(data_train[data_train['item_price'] > 100000].index, inplace=True)
print('Remove the outliers (item_cnt_day)')
data_train.drop(data_train[data_train['item_cnt_day'] > 1500].index, inplace=True)

# # plot pair
# print('plot pair and scatter')
# sns.pairplot(data_train[['date_block_num', 'shop_id', 'item_id', 'item_price', 'item_cnt_day']])
# plt.savefig('pair_new.png')

# plt.scatter(data_train['date'], data_train['item_price'])
# plt.savefig('scatter_new.png')

# # Drop columns
print('Drop column')
data_train.drop(['date_block_num'], axis=1, inplace=True)
print(data_train.head())

# Data train
print('Data train processing...')
print('groupby setting ...')
df_train = data_train.groupby(['date','shop_id','item_id']).sum() # setting weight
print(df_train.head())
df_train['item_price'].apply(lambda x: log10(x))
print('pivot_table setting ...')
df_train = df_train.pivot_table(index=['shop_id','item_id', 'item_price'], 
                                columns='date', 
                                values='item_cnt_day', 
                                fill_value=0)
print(df_train)