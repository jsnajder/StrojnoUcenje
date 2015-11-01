# Sveučilište u Zagrebu
# Fakultet elektrotehnike i računarstva
#
# Strojno ucenje 
# http://www.fer.hr/predmet/su
#
# (c) 2015 Jan Snajder

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode
        self.encoders ={}

    def fit(self,X,y=None):
        if self.columns is not None:
            for colname in self.columns:
                self.encoders[colname] = LabelEncoder().fit(X[colname])
        else:
            for colname,col in X.iteritems():
                self.encoders[colname] = LabelEncoder().fit(col)
        return self

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        for colname in self.encoders.keys():
            output[colname] = self.encoders[colname].transform(output[colname])
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

    def encoders(self):
        return self.encoders

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

class PolyRegression:

    def __init__(self, order):
        self.order = order
        self.h = LinearRegression()
        
    def fit(self, X, y): 
        Xt = PolynomialFeatures(self.order).fit_transform(X)
        self.h.fit(Xt, y)
        return self
    
    def predict(self, X):
        Xt = PolynomialFeatures(self.order).fit_transform(X)
        return self.h.predict(Xt)
    
    def __call__(self, x):
        return self.predict(x)[0]
