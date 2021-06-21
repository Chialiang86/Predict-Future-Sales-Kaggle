---
tags: DSAI, python
---

# Predict-Future-Sales Kaggle

- 詳細報告：[report link](https://docs.google.com/presentation/d/1nMD0G22-CNBULoP0P4Gtn9MAQsO_bTo0vI2hzNKGz1k/edit?usp=sharing)

## 執行方式

使用 pipenv 作為虛擬環境

- 啟動虛擬環境
```shell 
$ pipenv --python 3.8
```

- 虛擬環境建立、指定版本為 python 3.8
```shell 
$ pipenv shell
$ (Predict-Future-Sales-Kaggle) chialiang86@chialiang86-System-Product-Name:~/Desktop/2021_DSAI/Predict-Future-Sales-Kaggle$
```

- 套件安裝：附有 Pipfile, Pipfile.lock 檔案，直接安裝
```shell
$ pipenv install
```
:::info
[packages]
- numpy
- pandas
- seaborn
- matplotlib
- scikit-learn
- lightgbm

[requires]
- python_version = "3.8"
:::

### 執行順序
提供兩個 .py 檔
- preprocessing.py : 做feature engineering，產生 training.pkl, testing.pkl 檔
```shell
$ python preprocessing.py
```
執行完後會分別產生 training/testing 的 .pickle 檔及 .csv 檔 -> .pickle 為給 training.py 用

- training.py : 讀取 training.pkl, testing.pkl 進行訓練以及分析
```shell
$ python testing.py
```
執行後會產生 submission.csv

