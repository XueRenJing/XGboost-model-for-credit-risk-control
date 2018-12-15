# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 14:41:04 2018

@author: xujie
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

# =============================================================================
# sum-feature-score筛选变量
# =============================================================================
os.chdir(r'F:\data\xujie\scorecard\scorecard\TT05_D_xgb_180910')
datam=pd.read_pickle('D_ysrawdata.pkl')#datam存放已经加工好的变量，包括gid,y_label。

train=datam[datam['type']=='train']
test=datam[datam['type']=='test']
train_x=train.drop(['gid','y_label'],axis=1)#去掉y值和gid(也可根据数据实际情况处理，train_x中放的是可以直接入模的变量)
train_y=train.y
val_x=test.drop(['gid','y_label'],axis=1)
val_y=test.y


# =============================================================================
# #设定了随机参数，跑6次模型，最后会抛出6个模型的feature score文件
# =============================================================================
dtrain = xgb.DMatrix(train_x, label=train_y)
import random

def pipeline(iteration,random_seed,gamma,max_depth,lambd,subsample,colsample_bytree,min_child_weight,eta):
    params={
            'booster':'gbtree',
	    'objective': 'rank:pairwise',
	    'scale_pos_weight': float(len(train_y)-sum(train_y))/float(sum(train_y)),
	    'eval_metric': 'auc',
	    'gamma':gamma,
	    'max_depth':max_depth,
	    'lambda':lambd,
	    'subsample':subsample,
	    'colsample_bytree':colsample_bytree,
	    'min_child_weight':min_child_weight, 
	    'eta': eta,
	    'seed':random_seed
	}

    watchlist  = [(dtrain,'train')]
    model = xgb.train(params,dtrain,num_boost_round=300,evals=watchlist)
    feature_score = model.get_fscore()
    feature_score = sorted(feature_score.items(), key=lambda x:x[1],reverse=True)
    fs = []
    for (key,value) in feature_score:
        fs.append("{0},{1}\n".format(key,value))
    
    with open('feature_score_{0}.csv'.format(iteration),'w') as f:
        f.writelines("feature,score\n")
        f.writelines(fs)


if __name__ == "__main__":
    random_seed = list(range(10000,20000,100))
    gamma =list(range(1,60,1))
    max_depth = [3,4,5,6,7]
    lambd = list(range(1,30,1))
    subsample = [i/1000.0 for i in range(600,800,2)]
    colsample_bytree = [i/1000.0 for i in range(600,800,2)]
    min_child_weight = [i/10.0 for i in range(20,220,1)]
    eta = [i/100.0 for i in range(1,11,1)]

    random.shuffle(random_seed)
    random.shuffle(gamma)
    random.shuffle(max_depth)
    random.shuffle(lambd)
    random.shuffle(subsample)
    random.shuffle(colsample_bytree)
    random.shuffle(min_child_weight)
    random.shuffle(eta)

    for i in range(6):
        pipeline(i,random_seed[i],gamma[i],max_depth[i%3],lambd[i],subsample[i],colsample_bytree[i],min_child_weight[i],eta[i])
        
# =============================================================================
# #新建一个文件夹，把上一步得到的6个文件放到新的文件夹里面。这里会把之前6个模型feature score进行加总，得到重要的文件。    
# 
# =============================================================================
os.chdir(r'F:\data\xujie\scorecard\scorecard\54催收TTO5\TT05_D_xgb_180910\TT05_D_sumfeaturescore_180911')

files = os.listdir()
fs = {}
for f in files:
    t = pd.read_csv(f)
    t.index = t.feature
    t = t.drop(['feature'],axis=1)
    d = t.to_dict()['score']
    for key in d:
        if key in fs:
            fs[key] += d[key]
        else:
            fs[key] = d[key] 
        
fs = sorted(fs.items(), key=lambda x:x[1],reverse=True)

t = []
for (key,value) in fs:
    t.append("{0},{1}\n".format(key,value))

with open('TT05_D_sumfeaturescore_180911.csv','w') as f:
    f.writelines("feature,score\n")
    f.writelines(t)
