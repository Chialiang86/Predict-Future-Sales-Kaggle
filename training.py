import numpy as np
import pandas as pd

from pandas.core.frame import DataFrame
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingRegressor

# cols setting
cols = ['shop_id', 'item_id', 'item_category_id', 'city_code', 'avg_price']
# cols = []
i = 1
max_i = 34
while i <= max_i:
    y = 2013 + i // 12
    for m in range(1, 13):
        cols.append('{}-{}'.format(y, str(m).zfill(2)))
        i += 1
        if i > max_i:
            break
print(cols)

# Reading Data
df_train = pd.read_csv('training.csv', usecols=cols)
df_test = pd.read_csv('testing.csv', usecols=cols)

# X, Y setting
print('training data setting...')
X_train = df_train.drop(['2015-10'], axis = 1)
Y_train = df_train['2015-10'].values
print(X_train.shape, Y_train.shape)

print('testing data setting...')
x_test = df_test.drop(['2013-01'], axis=1)
print(x_test.shape)

# Split the data
print('spliting data...')
x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=0)
print(x_train.shape, y_train.shape)

# Gradient boosting Regression
print('boosting Regression...')

params = {'n_estimators': 200,
          'max_depth': 4,
          'learning_rate': 0.05,
          'verbose': 2}
gbr = GradientBoostingRegressor(**params)
gbr.fit(x_train, y_train)



# plt feature importance
print('ploting feature importance...')
feature_importance = gbr.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, np.array(x_train.columns)[sorted_idx])
plt.title('Feature Importance (MDI)')

result = permutation_importance(gbr, x_val, y_val, n_repeats=10,
                                random_state=42, n_jobs=2)
sorted_idx = result.importances_mean.argsort()
plt.subplot(1, 2, 2)
plt.boxplot(result.importances[sorted_idx].T,
            vert=False, labels=np.array(x_train.columns)[sorted_idx])
plt.title("Permutation Importance (test set)")
fig.tight_layout()
plt.savefig('feature_importances.png')


# plot prediction result
print('ploting predict result...')
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

# for i in range(len(diff_vdf)):
#     print('expected {}, get {}'.format(y_val[i], y_pre_v[i]))

thresh = 50

print('============= show gbr result =============')
print('Train set mse:', mean_squared_error(y_train, y_pre_t))
print('Val set mse:', mean_squared_error(y_val, y_pre_v))
print('data of large error in training (thresh = {}):'.format(thresh))
print(x_train.iloc[diff_tdf[abs(diff_tdf['training'].astype(int)) > thresh].index.tolist()])
print('data of large error in validation (thresh = {}):'.format(thresh))
print(x_val.iloc[diff_vdf[abs(diff_vdf['validation'].astype(int)) > thresh].index.tolist()])
print('===========================================\n')


# setting submission.csv
print('setting submission.csv ...')
pred_test = gbr.predict(x_test)

submission = pd.DataFrame({
    'ID':range(214200),
    'item_cnt_month':pred_test
})
submission.to_csv('submission.csv', index=False)
submission.head()

print('process complete.')