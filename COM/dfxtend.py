# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 12:44:16 2018
Functions for accessors of the pandas dataframe
@author: tangk
"""

import pandas as pd


@pd.api.extensions.register_dataframe_accessor('chdata')
class ChData(object):
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        self.t = None
        
    def get_features(self, feature_funs, t=None, rename=True):
        """ feature_funs is one of: 
            - list of functions used to get a feature
            - dict of {function name: callable function}
         returns features as a list or dict, depending on input
         if dict, then key names are by default appended to column names"""
        if isinstance(feature_funs,list):
            out = []
            for funs in feature_funs:
                features = self._obj
                for subfuns in funs:
                    features = features.applymap(subfuns)
                out.append(features)
                
        elif isinstance(feature_funs,dict):
            out = {}
            for funs in feature_funs:
                features = self._obj
                for subfuns in feature_funs[funs]:
                    features = features.applymap(subfuns)
                if rename:
                    features = features.rename(lambda x: funs + x,axis=1)
                out[funs] = features
        return out 
    
    def re_cutoff(self,cutoff):
        return self._obj.applymap(lambda x: x[cutoff])
    
#    @property
#    def t(self):
#        return self._t
        

@pd.api.extensions.register_dataframe_accessor('table')
class Table(object):
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
    
    def query_list(self, column,qlist):
        """ queries a dataframe by whether column contains qlist """
        df = self._obj
        return df[df[column].isin(qlist)]
    
    def names_to_se(self, names):
        """ return a dict {categorical names: corresponding tcs}
        names is a dict {categorical names: query parameters} """
        return {n: list(self._obj.query(names[n]).index) for n in names.keys()}