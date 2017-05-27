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



def main(overwrite):

	train = pd.read_csv('../input/train_u6lujuX_CVtuZ9i.csv')
	test = pd.read_csv('../input/test_Y3wMUE5_7gLdaTN.csv')
	test_id = test.Loan_ID

	y = train.Loan_Status

	mp = {'Y': 1, 'N': 0}
	y = y.map(mp)
	train.drop(['Loan_Status', 'Loan_ID'], axis=1, inplace=True)

	data = pd.concat([train, test])

	data = data.fillna(-1)
	data = pd.get_dummies(data)

	scaler = preprocessing.MinMaxScaler().fit(data)
	data = scaler.transform(data)

	x = data[:train.shape[0]]
	test = data[train.shape[0]:]

	return x, y, test_id, test        


    
