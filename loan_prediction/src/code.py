from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
import feature_engineering
from imp import reload
reload(feature_engineering)
import numpy as np
import os
import pandas as pd
import time
import xgboost as xgb
from scipy.stats import mode
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

n_splits = 5
final_accuracy = 0
random_state = 1971

X, y, test_id, test = feature_engineering.main2(overwrite=False)
print (X.shape)
# exit()
predictions = np.zeros((test.shape[0], n_splits))
loan_status = []

model = LogisticRegression()
# model = svm.SVC()
# model = KNeighborsClassifier(n_neighbors=3)
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.01, max_depth=6, random_state=0)


skf = StratifiedKFold(n_splits=n_splits, random_state=random_state)
for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
   X_train, x_valid = X[train_index, :], X[test_index, :]
   y_train, y_valid = y[train_index], y[test_index]
   model.fit(X_train, y_train)
   accuracy = accuracy_score(y_valid, model.predict(x_valid))   
   predictions[:, fold] = model.predict(test)   
   final_accuracy += accuracy
   print ('fold %d Accuracy %f'%(fold, accuracy))

final_accuracy /= float(n_splits)
predictions = mode(predictions, axis=1)[0].astype(int)

for p in predictions:
	p = 'Y'if p == 1 else 'N'
	loan_status.append(p)

submission = pd.DataFrame({'Loan_ID': test_id, 'Loan_Status': loan_status})
sub_file = '../output/%f__%s.csv'%(final_accuracy, str(time.ctime()).replace(' ', '_'))
submission.to_csv(sub_file, index=False)

sub_file = '../output/sample_submission.csv'
submission.to_csv(sub_file, index=False)
print ('Final loss %f'%final_accuracy)