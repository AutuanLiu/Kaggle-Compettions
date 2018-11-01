"""
Email: autuanliu@163.com
Date: 2018/11/01
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from xgboost import XGBRegressor

# 导入数据
# 导入数据
datadir = 'dataset/'
# 使用全部特征
train_data = pd.read_csv(f'{datadir}train_process.csv').values
test_data = pd.read_csv(f'{datadir}test_process.csv').values
# 使用部分特征 忽略相关特征
# train_data = pd.read_csv(f'{datadir}part_train_process.csv').values
# test_data = pd.read_csv(f'{datadir}part_test_process.csv').values
# 注意数据类型的转换
X_test = test_data[:, 1:].astype('float64')
id = test_data[:, 0].astype('int64')
X_train = train_data[:, 1:-1].astype('float64')
y_train = train_data[:, -1].astype('float64')

# 训练集验证集划分
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, test_size=0.3, random_state=0)

MLA = [
    GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt',
                              min_samples_leaf=15, min_samples_split=10, loss='huber', random_state=5),
    make_pipeline(RobustScaler(), ElasticNet(
        alpha=0.0005, l1_ratio=.9, random_state=3)),
    make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1)),
    RandomForestRegressor(n_estimators=20),
    XGBRegressor(olsample_bytree=0.4603, gamma=0.0468, learning_rate=0.05, max_depth=3, min_child_weight=1.7817,
                 n_estimators=2200, reg_alpha=0.4640, reg_lambda=0.8571, subsample=0.5213, silent=1, random_state=7, nthread=-1, verbose=False),
    CatBoostRegressor(iterations=500, learning_rate=0.1, depth=6, loss_function='RMSE', random_state=123,
                      leaf_estimation_method='Gradient', l2_leaf_reg=2, fold_len_multiplier=1.2, train_dir='log', verbose=False),
    LGBMRegressor(objective='regression', num_leaves=5, learning_rate=0.05, n_estimators=720, max_bin=55, bagging_fraction=0.8,
                  bagging_freq=5, feature_fraction=0.2319, feature_fraction_seed=9, bagging_seed=9, min_data_in_leaf=6, min_sum_hessian_in_leaf=11)
]


pred = 0
for m in MLA:
    m.fit(X_train, y_train)
    y_pred = m.predict(X_valid)
    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    pred += np.exp(m.predict(X_test))
    print(rmse)

# 模型预测结果
# xt = XGBRegressor()
# xt.fit(X_train, y_train)
# pred = np.exp(xt.predict(X_test))
pred /= 7
# 保存预测
preds = pd.DataFrame()
preds['Id'] = id.astype(np.int32)
preds['SalePrice'] = pred.astype(np.float32)
preds.to_csv(f'{datadir}submission.csv', index=False)
