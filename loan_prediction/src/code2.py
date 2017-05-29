import pandas as pd
from sklearn import svm
import time
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from scipy.stats import mode
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing


cat_cols = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
LE = preprocessing.LabelEncoder()
class missing_value_prediction:
	def main(self, col):
		train = pd.read_csv('../input/train.csv')
		test = pd.read_csv('../input/test.csv')

		train.drop('Loan_Status', axis=1, inplace=True)
		data = pd.concat([train, test], axis=0)
		data.drop('Loan_ID', axis=1, inplace=True)

		data = data[['Education', 'ApplicantIncome', 'CoapplicantIncome', 'Property_Area', col]]		
		
		
		target = data[col]
		y = data[col]		
		X = data.drop(col, axis=1)		
		X = pd.get_dummies(X)
				

		nan_indices = y[pd.isnull(y)].index
		y = y.drop(nan_indices)
		X = X.drop(nan_indices)

		X = (X - X.min())/(X.max()-X.min())
		X = X - X.mean()
		

		final_accuracy = 0
		n_splits = 10
		skf = StratifiedKFold(n_splits=n_splits, random_state=1971)		
		
		clf = LinearRegression() if col not in cat_cols else LogisticRegression()			
		

		X = X.as_matrix()
		y = y.as_matrix()
		if col in cat_cols:
			y = LE.fit_transform(y)
		predictions = np.zeros((X[nan_indices].shape[0], n_splits))

		for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
		   X_train, x_valid = X[train_index, :], X[test_index, :]
		   y_train, y_valid = y[train_index], y[test_index]
		   
		   clf.fit(X_train, y_train)
		   predictions[:, fold] = clf.predict(X[nan_indices])
		   if col not in cat_cols:
		   	accuracy = mean_squared_error(y_valid, clf.predict(x_valid))      
		   else:
		   	accuracy = accuracy_score(y_valid, clf.predict(x_valid))      
		   final_accuracy += accuracy
		   # print ('fold %d %s Accuracy %f'%(fold, col, accuracy))

		
		print '%s Accuracy %f'%(col, final_accuracy/float(n_splits))
		pred = predictions.mean(axis=1) if col not in cat_cols else mode(predictions, axis=1)[0].astype(int)				
		pred = LE.inverse_transform(pred) if col in cat_cols else pred
		target[target.isnull()]=pred
		return target



class feature_engineering:
	def impute(self, data):		
		CH = missing_value_prediction()
		for col in data.columns:			
			if data[col].isnull().sum() > 0:				
					data[col] = CH.main(col)	
					
		# exit()
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
		data['f1'] = data.ApplicantIncome + data.CoapplicantIncome
		data['f2'] = data.LoanAmount/data.Loan_Amount_Term
		data['f3'] = data['f1']/data.Loan_Amount_Term
		data['f4'] = data.Married + '_' + data.Self_Employed
		
		
		# data['f4'] = (train.Property_Area == 'Semiurban').apply(int)
		# data['f5'] = (train.Education == 'Graduate').apply(int)
		# data['f6'] = (train.Married == 'Yes').apply(int)

		data = pd.get_dummies(data)
		
		for col in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'f1', 'f2', 'f3']:
			data[col] = (1.0 + data[col]).apply(np.sqrt)		
			# data[col] = (data[col] - data[col].min())/(data[col].max()-data[col].min())
			pass
		

		print data.head()

		x = data[:train.shape[0]]
		test = data[train.shape[0]:]

		return x.as_matrix(), y, test_id, test       

FE = feature_engineering()


n_splits = 5
final_accuracy = 0
random_state = 1971

X, y, test_id, test = FE.main()

predictions = np.zeros((test.shape[0], n_splits))
loan_status = []

model = LogisticRegression()


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