import pandas as pd
from sklearn import svm
import time
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from scipy.stats import mode

class Credit_History:
	def main(self):
		train = pd.read_csv('../input/train.csv')
		test = pd.read_csv('../input/test.csv')

		train.drop('Loan_Status', axis=1, inplace=True)
		data = pd.concat([train, test], axis=0)
		data.drop('Loan_ID', axis=1, inplace=True)

		data = data[['Education', 'ApplicantIncome', 'CoapplicantIncome', 'Property_Area', 'Credit_History']]
		full_credit_history = data.Credit_History
		y = data.Credit_History
		X = data.drop('Credit_History', axis=1)

		def impute(data):	
			for col in data.columns:		
				data[col] = data[col].fillna('empty') if data[col].dtype == 'object' \
							else data[col].fillna(data[col].mean())
			return data

		X = impute(X)
		X = pd.get_dummies(X)

		nan_indices = y[pd.isnull(y)].index
		y = y.drop(nan_indices)
		X = X.drop(nan_indices)

		final_accuracy = 0
		n_splits = 5
		skf = StratifiedKFold(n_splits=n_splits, random_state=1971)
		clf = GradientBoostingClassifier()

		X = X.as_matrix()
		y = y.as_matrix()

		for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
		   X_train, x_valid = X[train_index, :], X[test_index, :]
		   y_train, y_valid = y[train_index], y[test_index]
		   clf.fit(X_train, y_train)
		   accuracy = accuracy_score(y_valid, clf.predict(x_valid))      
		   final_accuracy += accuracy
		   print ('fold %d Credit History Accuracy %f'%(fold, accuracy))


		clf.fit(X, y)
		pred = clf.predict(X[nan_indices])
		full_credit_history[full_credit_history.isnull()]=pred
		return full_credit_history



class feature_engineering:
	def impute(self, data):		
		for col in data.columns:				
			if col == 'Credit_History':
				CH = Credit_History()
				data[col] = CH.main()				
			else:	
				data[col] = data[col].fillna('empty') if data[col].dtype == 'object' \
							else data[col].fillna(data[col].mean())
		return data

	def main(self):
		train = pd.read_csv('../input/train.csv')
		test = pd.read_csv('../input/test.csv')
		test_id = test.Loan_ID

		y = train.Loan_Status

		mp = {'Y': 1, 'N': 0}
		y = y.map(mp)
		train.drop(['Loan_Status'], axis=1, inplace=True)

		data = pd.concat([train, test])
		data.drop(['Loan_ID'], axis=1, inplace=True)

		data = self.impute(data)	
		data = pd.get_dummies(data)
		
		x = data[:train.shape[0]]
		test = data[train.shape[0]:]

		return x.as_matrix(), y, test_id, test       

FE = feature_engineering()


n_splits = 5
final_accuracy = 0
random_state = 1971

X, y, test_id, test = FE.main()
print (X.shape)
# exit()
predictions = np.zeros((test.shape[0], n_splits))
loan_status = []

model = LogisticRegression()
# model = svm.SVC()
# model = KNeighborsClassifier(n_neighbors=3)
# model = GradientBoostingClassifier(n_estimators=250, learning_rate=0.05, max_depth=8, random_state=0)


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