'''
This demo is based on Kaggle Contest "Prudential Life Insurance Assessment"
which uses Quadratic Weighted Kappa for evaluation criterion.
(https://www.kaggle.com/c/prudential-life-insurance-assessment)

Before running, first download data to "../data" folder.
(https://www.kaggle.com/c/prudential-life-insurance-assessment/data)

Tested with Python2.7
'''

import xgboost as xgb
import pandas as pd 
import numpy as np 
from ml_metrics import quadratic_weighted_kappa

# global variables
split = 10
columns_to_drop = ['Id', 'Response']
xgb_num_rounds = 10000
num_classes = 8

# seed random
rand_seed = np.random.randint(10000000)
rand_seed = 955475
print('rand_seed=', rand_seed)
np.random.seed( rand_seed )

# eta schedule
eta_list = [0.06] * 150
eta_list = eta_list + [0.05] * 150 
eta_list = eta_list + [0.04] * 300 

def get_kappa_params():
    params = {}
    params["objective"] = "reg:kappa"     
    params["eval_metric"] = "kappa"
    params["booster"] = "gbtree"
    params["subsample"] = 0.9
    params["colsample_bytree"] = 0.3
    params["max_depth"] = 7
    params["min_child_weight"] = 5
    params["silent"] = 0
    params["early_stopping_rounds"] = 100
    #params["x"] = "5,0,1"  ## default values
    plst = list(params.items())
    return plst, params

def get_params():
    return get_kappa_params()

def eval_wrapper(yhat, y):  
    y = np.array(y)
    y = y.astype(int)
    yhat = np.array(yhat)
    yhat = np.clip(np.round(yhat), np.min(y), np.max(y)).astype(int)   
    return quadratic_weighted_kappa(yhat, y)

print("load the data using pandas")
train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")
data = train

# combine train and test
data = data.append(test)

# create any new variables    
data['Product_Info_2_char'] = data.Product_Info_2.str[1]
data['Product_Info_2_num'] = data.Product_Info_2.str[2]

# factorize categorical variables
data['Product_Info_2'] = pd.factorize(data['Product_Info_2'])[0]
data['Product_Info_2_char'] = pd.factorize(data['Product_Info_2_char'])[0]
data['Product_Info_2_num'] = pd.factorize(data['Product_Info_2_num'])[0]

print('eliminate missing values')    
data.fillna(-1, inplace=True)

# Provide split column
data['Split'] = np.random.randint(split, size=data.shape[0]).astype('int64')

# split off true test data
test = data[data['Response']<1].copy()
data = data[data['Response']>0].copy()

# fix the dtype on the label column
y = data['Response'].astype(float)
data['Response'] = y

# split train and valid
all = data
train = data[data['Split']>0].copy()
valid = data[data['Split']==0].copy()
y_all = all['Response'].astype(int)
y_train = train['Response'].astype(int)
y_valid = valid['Response'].astype(int)

# convert data to xgb data structure
xgtrain = xgb.DMatrix(train.drop(columns_to_drop, axis=1), train['Response'].values)
xgvalid = xgb.DMatrix(valid.drop(columns_to_drop, axis=1), valid['Response'].values)  

# get the parameters for xgboost
plst, params = get_params()
print(plst)      

esr = None
if "early_stopping_rounds" in params:
    esr = params["early_stopping_rounds"] 

# train model
watchlist  = [(xgtrain,'train'), (xgvalid,'eval')]
model = xgb.train(plst, xgtrain, xgb_num_rounds, watchlist, learning_rates=eta_list, early_stopping_rounds=esr)
best_iter = model.best_iteration

# get preds
train_preds = model.predict(xgtrain, ntree_limit=best_iter)
valid_preds = model.predict(xgvalid, ntree_limit=best_iter)

print('train score is:', eval_wrapper(train_preds, y_train))
print('valid score is:', eval_wrapper(valid_preds, y_valid))

# retrain on train+valid data
y_train = y_all
xgdata = xgb.DMatrix(all.drop(columns_to_drop, axis=1), all['Response'].values)
xgtest = xgb.DMatrix(test.drop(columns_to_drop, axis=1), label=test['Response'].values)
watchlist  = [(xgdata,'train')]
model = xgb.train(plst, xgdata, best_iter+50, watchlist, learning_rates=eta_list)
train_preds = model.predict(xgdata, ntree_limit=best_iter)
print('train score is:', eval_wrapper(train_preds, y_train))

# dump model
#model.dump_model('../data/xgb.model', with_stats=True)

# generate predictions on test data
print('generating predictions on test data')
test_preds = model.predict(xgtest, ntree_limit=best_iter)
final_test_preds = np.round(np.clip(test_preds, 1, 8)).astype(int)
preds_out = pd.DataFrame({"Id": test['Id'].values, "Response": final_test_preds})
preds_out = preds_out.set_index('Id')
preds_out.to_csv('../data/test_preds.csv')

print('Done!')
