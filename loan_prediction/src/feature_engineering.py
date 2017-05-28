# -*- coding: utf-8 -*-
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.cross_validation import train_test_split
from string import punctuation
import datetime
import nltk
import numpy as np
import operator, os
import pandas as pd
import re
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA


def impute(data):
	for col in data.columns:		
		data[col] = data[col].fillna('empty') if data[col].dtype == 'object' \
					else data[col].fillna(data[col].mean())
	return data


def main2(overwrite):

	train = pd.read_csv('../input/train.csv')
	test = pd.read_csv('../input/test.csv')
	test_id = test.Loan_ID

	y = train.Loan_Status

	mp = {'Y': 1, 'N': 0}
	y = y.map(mp)
	train.drop(['Loan_Status'], axis=1, inplace=True)

	data = pd.concat([train, test])
	data.drop(['Loan_ID'], axis=1, inplace=True)

	data = impute(data)	

	features = pd.DataFrame()
	features['Credit_History'] = data['Credit_History']
	features['Property_Area'] = data['Property_Area']
	features['LoanAmount/Loan_Amount_Term'] = data['LoanAmount']/data['Loan_Amount_Term']
	
	data = pd.get_dummies(features)

	print (data.head())



	x = data[:train.shape[0]]
	test = data[train.shape[0]:]

	return x.as_matrix(), y, test_id, test       

def main(overwrite):

	train = pd.read_csv('../input/train.csv')
	test = pd.read_csv('../input/test.csv')
	test_id = test.Loan_ID

	y = train.Loan_Status

	mp = {'Y': 1, 'N': 0}
	y = y.map(mp)
	train.drop(['Loan_Status'], axis=1, inplace=True)

	data = pd.concat([train, test])
	data.drop(['Loan_ID'], axis=1, inplace=True)

	data = impute(data)	
	data = pd.get_dummies(data)

	scaler = preprocessing.MinMaxScaler().fit(data)
	data = scaler.transform(data)

	x = data[:train.shape[0]]
	test = data[train.shape[0]:]

	return x, y, test_id, test       