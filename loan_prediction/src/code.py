from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
import feature_engineering
import numpy as np
import os
import pandas as pd
import time
import xgboost as xgb
from imblearn.combine import SMOTEENN
from sklearn import preprocessing

n_splits = 5
n_rounds = 5000
random_seed = 1971

X, y, test_id, test = feature_engineering.main(overwrite=False)

predictions = np.zeros((test.shape[0], n_splits))
test  = xgb.DMatrix(test)


losses = 0

params = {}

params['eta'] = 0.1
params['eval_metric'] = ['error']
params['max_depth'] = 12
params['n_thread'] = 4
params['objective'] = 'binary:logistic'
params['seed'] = random_seed
params['silent'] = 1


def train_xgb(x_train, y_train, x_valid, y_valid, params):		
	# params['scale_pos'] = np.count_nonzero(y_train==0)/np.count_nonzero(y_train==1)
	xg_train = xgb.DMatrix(x_train, label=y_train)
	xg_val = xgb.DMatrix(x_valid, label=y_valid)
	watchlist  = [(xg_train,'train'), (xg_val,'eval')]
	
	model = xgb.train(params, xg_train, n_rounds, watchlist, verbose_eval=True, early_stopping_rounds=500) 
	pred = model.predict(xgb.DMatrix(x_valid), ntree_limit=model.best_ntree_limit)	
	loss = model.best_score	

	return model, loss



if n_splits == 5:
	skf = StratifiedKFold(n_splits=n_splits, random_state=random_seed)
	for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
	   X_train, x_valid = X[train_index, :], X[test_index, :]
	   y_train, y_valid = y[train_index], y[test_index]
	   model, loss = train_xgb(X_train, y_train, x_valid, y_valid, params)
	   predictions[:, fold] = model.predict(test)
	   losses += loss
	   print ('fold %d Loss %f'%(fold, loss))
else:
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=random_seed, stratify=y)
	fold = 0
	model, loss = train_xgb(X_train, y_train, X_test, y_test, params)
	predictions[:, fold] = model.predict(test)
	losses += loss



prediction = (predictions.mean(axis=1)+0.5).astype(int)
loan_status = []
for p in prediction:
	if p == 1:
		loan_status.append('Y')
	else:
		loan_status.append('N')
loss = losses/float(n_splits)

submission = pd.DataFrame({'Loan_ID': test_id, 'Loan_Status': loan_status})
sub_file = '../output/%f__%s.csv'%(loss, str(time.ctime()).replace(' ', '_'))
submission.to_csv(sub_file, index=False)

sub_file = '../output/sample_submission.csv'
submission.to_csv(sub_file, index=False)
print ('Final loss %f'%loss)

# cmd = '7z a ../output/submission.7z %s'%sub_file
# os.system(cmd)

