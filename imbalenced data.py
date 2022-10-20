# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 11:29:04 2022

@author: kiran
"""

import pandas as pd
import seaborn as sns
import imblearn
import matplotlib.pyplot as mpl
data = pd.read_csv("C:/Users/kiran/Downloads/creditcard.csv")
data.columns
data['Class'].value_counts()
sns.countplot(data['Class'])

#under sampling

x= data.drop(labels=['Class'],axis=1)
x.columns
y= data.drop(labels=['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9','V10','V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20','V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'],axis=1)
y.columns 

#Ramdomly removing
from imblearn.under_sampling import RandomUnderSampler
random = RandomUnderSampler() 
x_res,y_res = random.fit_resample(x,y)
y_res.value_counts()
sns.countplot(y_res['Class'])
sns.countplot(data['Class'])

#Removing Nearest 
from imblearn.under_sampling import EditedNearestNeighbours
enn = EditedNearestNeighbours()
a,b= enn.fit_resample(x,y)
sns.countplot(data['Class'])
b['Class'].value_counts()
b.columns
sns.countplot(b['Class'])


#over sampler
#Random over sampler
from imblearn.over_sampling import RandomOverSampler
ros= RandomOverSampler()
b,c=ros.fit_resample(x,y)
sns.countplot(c['Class'])
sns.counplot(data['Class'])

#SMOTE
from imblearn.over_sampling import SMOTE
i,j = SMOTE().fit_resample(x,y)
sns.countplot(j['Class'])
sns.counplot(data['Class'])

#MIX OF BOTH OVER SAMPLING AND UNDER SAMPLING
from imblearn.combine import SMOTEENN
o,k=SMOTEENN().fit_resample(x,y)
sns.countplot(k['Class'])
