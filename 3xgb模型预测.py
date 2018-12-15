# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 15:33:18 2018

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

def ks_xgb(preds, train_data):
    true = train_data.get_label()
    fpr, tpr, thre=roc_curve(true, preds, pos_label=1)
    return 'ks', abs(fpr-tpr).max()

def ks_score(true, preds):
    fpr, tpr, thre=roc_curve(true, preds, pos_label=1)
    return abs(fpr-tpr).max()


#pred
os.chdir(r'F:\data\xujie\scorecard\scorecard\54催收TTO5\TT05_D_xgb_180910')
datan=pd.read_pickle('TT05_D_ysrawdata.pkl')
train=datan[datan['type']=='train']
test=datan[datan['type']=='test']

col2=pd.read_excel('TT05_D_sumfeaturescore_180911.xlsx',sheetname='col2')


test_out=datan

testout_x=test_out[col2['feature']]
dtestout = xgb.DMatrix(testout_x)
model=joblib.load('TT05P1_D_xgbmodel_180912.ml')
testout_pred = model.predict(dtestout)
testo_Idx = test_out[['gid','y','type']]
testo_result = pd.DataFrame(testo_Idx,columns=['gid','y','type'])
testo_result["p"] = testout_pred
testo_result["y"] =  test_out['y']
testo_result["p_ml"]=testo_result["p"].apply(lambda x:round(x,9))
testo_result["score"] =  testo_result['p'].apply(to_scoretz)
print(roc_auc_score(testo_result["y"],testo_result["p"]))
print(ks_score(testo_result["y"],testo_result["p"]))
0.9092856671978676
0.6610147966824065
testo_result.to_excel('TT05P1_D_xgbmlpredscore_180912.xlsx',index=False)













