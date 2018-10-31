"""
Email: autuanliu@163.com
Date: 2018/10/29
"""

import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from dnn import MLP, Titanic
from torch.utils.data import DataLoader
from collections import Counter

import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

# 导入数据
datadir = 'dataset/'
# 使用全部特征
train_data = pd.read_csv(f'{datadir}train_process.csv').values
test_data = pd.read_csv(f'{datadir}test_process.csv').values
# 使用部分特征 忽略name相关特征
# train_data = pd.read_csv(f'{datadir}part_train_process.csv').values
# test_data = pd.read_csv(f'{datadir}part_test_process.csv').values
# 注意数据类型的转换
X_test = test_data[:, 1:].astype('float32')
id = test_data[:, 0].astype('int32')
X_train = train_data[:, 2:].astype('float32')
y_train = train_data[:, 1].astype('int32')

# 训练集验证集划分
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=0)

# Kfold-cv
kf = KFold(n_splits=10, shuffle=True, random_state=123)

# 1 构建模型 逻辑回归
clf = LogisticRegression(random_state=0, max_iter=100)
clf.fit(X_train, y_train)
pred_val = clf.predict(X_valid)
acc = acc = accuracy_score(y_valid, pred_val)
print(f'LogisticRegression accuracy is {acc}.')

# 2 构造模型 xgboost
param = {
    "max_depth": 6, 
    "learning_rate": 0.1, 
    "n_estimators": 500, 
    "silent": True, 
    "objective": 'binary:logistic', 
    "booster": 'gbtree'
}
xgbc = xgb.XGBClassifier(**param)

# 训练模型
xgbc.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="auc", eval_set=[(X_valid, y_valid)], verbose=False)

# cv
for train_index, test_index in kf.split(X_train):
    xgb_model = xgbc.fit(X_train[train_index], y_train[train_index])
    predictions = xgb_model.predict(X_train[test_index])
    actuals = y_train[test_index]
    # print(confusion_matrix(actuals, predictions))

# 模型预测
pred_val = xgbc.predict(X_valid)
acc = accuracy_score(y_valid, pred_val)
print(f'XGBClassifier accuracy is {acc}.')

# 预测结果
pred = xgbc.predict(X_test)

# 3 构建模型 catboost
config = {
    'iterations': 500,
    'learning_rate': 0.1,
    'depth': 6,
    'loss_function': 'CrossEntropy',
    'eval_metric': 'Accuracy',
    'random_state': 123,
    'leaf_estimation_method': 'Gradient',
    'l2_leaf_reg': 2,
    'fold_len_multiplier': 1.2,
    'od_type': 'IncToDec',
    'train_dir': 'log'
}

# unpacking 的形式传入参数
model = CatBoostClassifier(**config)

# train
model.fit(X_train, y_train, use_best_model=True, eval_set=[(X_valid, y_valid)], verbose=False, early_stopping_rounds=10)

# make the prediction using the resulting model
preds_class = model.predict(X_valid, prediction_type='Class')
score = model.score(X_valid, y_valid)
print(f'CatBoostClassifier accuracy is {score}')

# 4 lightGBM
# 参数设置
params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'n_estimators': 150,
    'random_state': 123,
    'objective': 'binary',
    'num_leaves': 31,
    'learning_rate': 0.1,
}
gbm = LGBMClassifier(**params)

# 使用部分数据
# 训练
gbm.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric='logloss', early_stopping_rounds=15, verbose=False)

# predict
y_pred = gbm.predict(X_valid, num_iteration=gbm.best_iteration_)
print(f'LGBMClassifier accuracy is {accuracy_score(y_valid, y_pred)}')

# 5 MLP
dev = torch.device('cuda')
net = MLP(17).to(dev)
criterion = nn.BCELoss()
opt = optim.Adam(net.parameters(), lr=0.001)
lr_deacy = optim.lr_scheduler.StepLR(opt, 20)
num_epoch = 100
# dataloader 构造
def make_loader(X, y, sf=False):
    datasets = Titanic(X, y)
    return DataLoader(datasets, batch_size=16, shuffle=sf)
train_loader = make_loader(X_train, y_train, sf=True)
# train valid
for epoch in range(num_epoch):
    # train
    net.train()
    lr_deacy.step()
    for X, y in train_loader:
        X, y = X.to(dev), y.to(dev).float()
        pred = net(X)
        loss = criterion(pred.view_as(y), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    # print(f'epoch: {epoch+1}, --> train loss: {loss.item():.4f}')
with torch.no_grad():
    pred1 = net(torch.from_numpy(X_valid).cuda()).cpu().numpy()
pred1 = np.round(pred1)
acc = accuracy_score(y_valid, pred1)
print(f'MLP accuracy: {acc}')

# 6 MLPClass
cmlp = MLPClassifier(hidden_layer_sizes=(10))
cmlp.fit(X_train, y_train)
y_pred = cmlp.predict(X_valid)
acc = accuracy_score(y_valid, y_pred)
print(f'sklearn MLP accuracy: {acc}')

# 7 Randomforest
crf = RandomForestClassifier(n_estimators=20)
crf.fit(X_train, y_train)
y_pred = crf.predict(X_valid)
acc = accuracy_score(y_valid, y_pred)
print(f'sklearn RF tree accuracy: {acc}')

# 保存预测结果
# 记录 各个模型的预测结果用作最后的ensemble
ensemble = []
pred = clf.predict(X_test)  # 使用逻辑回归
ensemble += [pred.reshape(-1,)]
pred = xgbc.predict(X_test) # 使用xgboost模型
ensemble += [pred.reshape(-1,)]
pred = model.predict(X_test, prediction_type='Class')  # 使用catboost 
ensemble += [pred.reshape(-1,)]
pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)  # 使用lightGBM模型
ensemble += [pred.reshape(-1,)]
with torch.no_grad():
    pred = net(torch.from_numpy(X_test).cuda()).cpu().numpy()  # 使用MLP
pred = np.round(pred)
ensemble += [pred.reshape(-1,)]
pred = cmlp.predict(X_test)  # 使用 MLP C
ensemble += [pred.reshape(-1,)]
pred = crf.predict(X_test)  # 使用 RF
ensemble += [pred.reshape(-1,)]

# # voting
ensemble = np.stack(ensemble, axis=1)
lenx = ensemble.shape[0]
pred = np.array([Counter(ensemble[idx]).most_common(1)[0][0] for idx in range(lenx)])

preds = pd.DataFrame(dtype='int32')
preds['PassengerId'] = id.astype(np.int32)
preds['Survived'] = pred.astype(np.int32)
preds.to_csv(f'{datadir}submission.csv', index=False)

# check
sub_data = pd.read_csv(f'{datadir}submission.csv').values
