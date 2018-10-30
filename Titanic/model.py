"""
Email: autuanliu@163.com
Date: 2018/10/29
"""

import pandas as pd
import numpy as np

import xgboost as xgb
from sklearn.linear_model import LogisticRegression

# 导入数据
datadir = 'dataset/'
train_data = pd.read_csv(f'{datadir}train_process.csv').values
test_data = pd.read_csv(f'{datadir}test_process.csv').values
X_train = train_data[:, 2:]
y_train = train_data[:, 1]

X_test = test_data[:, 1:]
id = test_data[:, 0]

# 构造模型
xgbc = xgb.XGBClassifier()

# 训练模型
xgbc.fit(X_train, y_train)

# 预测结果
pred = xgbc.predict(X_test)

# 保存预测结果
preds = pd.DataFrame()
preds['PassengerId'] = id.astype(np.int32)
preds['Survived'] = pred.astype(np.int32)
preds.to_csv(f'{datadir}submission.csv', index=False)
