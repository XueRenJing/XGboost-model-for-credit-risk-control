# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 15:04:07 2018

@author: xuexue
"""
import os
import numpy as np
import pandas as pd
from datetime import date
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.externals import joblib
import random

def ks_xgb(preds, train_data):
    true = train_data.get_label()
    fpr, tpr, thre=roc_curve(true, preds, pos_label=1)
    return 'ks', abs(fpr-tpr).max()

def ks_score(true, preds):
    fpr, tpr, thre=roc_curve(true, preds, pos_label=1)
    return abs(fpr-tpr).max()

# =============================================================================
# xgb模型
# =============================================================================
os.chdir(r'F:\data\xujie\scorecard\scorecard\54催收TTO5\TT05_D_xgb_180910')
datan=pd.read_pickle('TT05_D_ysrawdata.pkl')
train=datan[datan['type']=='train']
test=datan[datan['type']=='test']
#第一步筛选出的变量
col1=pd.read_excel('TT05_D_sumfeaturescore_180911.xlsx')


# =============================================================================
# 变量多的时候，可以根据变量相关性，把想性高的变量提出。根据实际情况决定。
# =============================================================================
dcor=datan[col1['feature']]
cor=dcor.corr()
corr_cols = cor.columns
del_cols = []
for i in range(len(corr_cols)):
    for j in range(i+1,len(corr_cols)):
        if cor.loc[corr_cols[i],corr_cols[j]] >=0.9:
            del_cols.append(corr_cols[j])
list1=[item for item in corr_cols.tolist() if item not in del_cols]
col2 = pd.DataFrame(list1,columns=['feature'])

col2=pd.read_excel('TT05_D_sumfeaturescore_180911.xlsx',sheetname='col2')
train_x=train[col2['feature']]
train_y=train.y
val_x=test[col2['feature']]
val_y=test.y

# =============================================================================
# 
# =============================================================================
print('训练测试集总体bad_rate：',(datan.y.sum())/datan.y.count())
print('训练集坏样本量：',train_y.sum())
print('训练集好样本量：',train_y.count()-train_y.sum())
print('训练集总样本量：',train_y.count())
print('训练集bad_rate：',train_y.sum()/train_y.count())
print('测试集坏样本量：',val_y.sum())
print('测试集好样本量：',val_y.count()-val_y.sum())
print('测试集总样本量：',val_y.count())
print('测试集bad_rate：',val_y.sum()/val_y.count())
datan.groupby('y')['gid'].count()
#
0.3223277909738717
2859
5971
8830
0.32378255945639867
1212
2588
3800
0.31894736842105265
Out[126]: 
y
0    8559
1    4071


#调参
dtrain = xgb.DMatrix(train_x, label=train_y)
dval = xgb.DMatrix(val_x, label=val_y)
watchlist  = [(dtrain,'train'),(dval,'val')]

params={
    'booster':'gbtree',
	'objective': 'binary:logistic',
	'eval_metric': 'auc',
    'max_depth':3,
    'min_child_weight':12, 
	'gamma':5,
	'subsample':0.75,
	'colsample_bytree':0.88,
    'alpha':8,
    'lambda':2,
	'eta': 0.08,
    'scale_pos_weight':1,
	'seed':0,
    'silent':0
      }
#训练测试集的AUC
xgb.train(params,dtrain,num_boost_round=1500,evals=watchlist,early_stopping_rounds=100) 

#训练测试集的KS
xgb.train(params,dtrain,num_boost_round=266,evals=watchlist,feval=ks_xgb,maximize=False)

#训练测试集的CV5_AUC
cvauc=xgb.cv(params,dtrain,num_boost_round=266,nfold=5,metrics='auc',early_stopping_rounds=100,verbose_eval=True)

#训练测试集的CV5_KS
cvks=xgb.cv(params,dtrain,num_boost_round=266,nfold=5,metrics='auc',verbose_eval=True,feval=ks_xgb)



# =============================================================================
# 模型训练好后，把模型保存出来
# =============================================================================
model=xgb.train(params,dtrain,num_boost_round=266,evals=watchlist)
joblib.dump(model,'TT05P1_D_xgbmodel_180912.ml')
model.dump_model('TT05P1_D_xgbmodel_180912.txt')


#feature-score
feature_score = model.get_fscore()
feature_score = sorted(feature_score.items(), key=lambda x:x[1],reverse=True)
fs=pd.DataFrame(feature_score,columns=['Type','Value'])
fs.to_csv('TT05P1_D_xgbfeaturescore_180912.csv',index=None)






















