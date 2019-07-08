# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:09:10 2019

helper functions for linear models

@author: tangk
"""
import pandas as pd
import numpy as np

from collections.abc import Iterable
from sklearn.preprocessing import StandardScaler
        
def preprocess_data(x, y, treat_nan='drop', scale=True):
    """x and y are either pandas Series or dataframes and arethe inputs and 
    outputs of your model, respectively. Returns x and y after the following 
    preprocessing steps:
        1. converts both to DataFrames if not already 
        2. drops all columns from x that can be found in y
        3. treats missing values in one of the following ways:
            'drop':  finds all samples where nans are found in either x or y and
                     drops them from both x and y
            'dropx': finds all samples where nans are found in x and
                     drops them from both x and y
            'mean':  finds all samples in both x and y where nans are found and
                     replaces them with the mean values of those columns
            'meanx': finds all samples in x only where nans are found and replaces
                     them with the mean values of those columns
            'none':  does nothing
    """
    if len(x.shape)==1:
        x = x.to_frame()
    if len(y.shape)==1:
        y = y.to_frame()
        
    x = x.drop([i for i in y.columns if i in x.columns], axis=1)
    
    if treat_nan=='drop':
        drop = x.loc[x.isna().any(axis=1)].append(y.loc[y.isna().any(axis=1)]).index
        x = x.drop(drop)
        y = y.drop(drop)
    elif treat_nan=='dropx':
        drop = x.loc[x.isna().any(axis=1)].index
        x = x.drop(drop)
        y = y.drop(drop)
    elif treat_nan=='mean':
        for col in x:
            x[col] = x[col].replace(np.nan, x[col].mean())
        for col in y:
            y[col] = y[col].replace(np.nan, y[col].mean())
    elif treat_nan=='meanx':
        for col in x:
            x[col] = x[col].replace(np.nan, x[col].mean())
    
    if scale:
        ss = StandardScaler()
        x = pd.DataFrame(ss.fit_transform(x), columns=x.columns, index=x.index)
    return x, y

def drop_correlated(x,y, thresh=0.5):
    """drops the columns of x that are correlated with the columns of y with a
    threshold of thresh"""
    corr = pd.concat((x,y), axis=1).corr().abs().loc[x.columns, y.columns]
    keep = corr.loc[x.columns, y.columns]<=thresh
    x = x[keep.index[keep.all(axis=1)]]
    return x

class VariableSelector(object):
    """an object of class VariableSelector takes inputs/outputs x and y and 
    selects the columns of x that best predict y using an iterative method"""
    def __init__(self, x, y, model, predictors=set(), incr_thresh=0.03, corr_thresh=0.4, eval_model=None):
        """initializes the input/output data, the predictors, the columns
        to drop during the iteration, the score, the function used to find 
        predictors of y, and the correlation matrix. *args and **kwargs are passed
        during initialization of the model.
        
        x, y: dataframes of input/output data
        model: either an instantiated model (the same model with the same 
               parameters is used every time) or an iterable where a different 
               model is called every time. Both must have a fit method and the 
               coefficients stored after fitting must be in the model.coef_ 
               attribute
        predictors: pre-specified known list of predictors 
        incr_thresh: threshold for improvement of score
        eval_model: optional model used to evaluate the fit
        """
        
        self.x = x # original x values
        self._test_x = x # x values left to try
        self.y = y
        self.predictors = predictors 
        self.incr_thresh = incr_thresh
        self.model = model
        self.eval_model = eval_model
        self.corr_thresh = corr_thresh

        self.corr = x.corr().abs()
#        if len(predictors)>0:
#            self.score = self._evaluate_model_fit(model, self.predictors)
#        else:
#            self.score = 0
        self.score = 0 # fix this later
    
    def find_variables(self):
        """
        computes the following steps:
            1. get candidate columns using an instance of the model
            2. evaluates the fit of the model, by default using the fit_model.score
               but optionally using eval_model.score() (preference for eval_model
               over fit_model). whichever eval method used must have a fit and a
               score method.
            3. if the score improves, adds the identified predictor to self.predictors
               and drops it from self._test_x
            4. gets a list of all columns that are correlated with the newly
               selected columns and drops them from self._test_x
        details on scoring:
            if using self.model, then the model must have a fit and a score 
            if using eval_model, then the model must have a fit and a score
        """
        continue_run = 1
        while continue_run:
            if isinstance(self.model, Iterable):
                model = next(self.model)
            else:
                model = self.model
    
            candidate_cols = self._find_candidate_columns(model)    
            if len(candidate_cols)==0:
                print('No candidate columns found!')
                break
            score = self._evaluate_model_fit(model, candidate_cols)
            continue_run = self._maybe_update_predictors(candidate_cols, score)
        print('found variables', self.predictors)
    
    def _find_candidate_columns(self, model):        
        model = model.fit(self._test_x, self.y)
        coefs = pd.Series(np.squeeze(model.coef_), index=self._test_x.columns)
        candidate_cols = set(coefs[coefs.abs()>0].index)
        return candidate_cols
    
    def _evaluate_model_fit(self, model, candidate_cols):
        y = self.y
        x_cols = self.predictors.union(candidate_cols)
        x = self.x[list(x_cols)]
        if self.eval_model is not None:
            eval_model = self.eval_model.fit(x, y)
            score = eval_model.score(x, y)
        else:
            model = model.fit(x, y)
            score = model.score(x, y)
        return score
    
    def _maybe_update_predictors(self, candidate_cols, score):
        thresh = self.incr_thresh
        predictors = self.predictors.union(candidate_cols)
        corr_thresh = self.corr_thresh
        if score==0:
            print('Error: model with predictors {0} gives a score of 0.'.format(predictors))
            return 0
        elif score - self.score < thresh:
            print('For predictors {0}: improvement in score {1} is below the threshold of {2}.'.format(predictors, score, thresh))
            return 0
        print('Predictors: {0}. Score: {1}. Adding columns {2}.'.format(predictors, score, candidate_cols))
        self.score = score
        self.predictors = predictors
        
        drop_cols = self.corr[list(predictors)]
        drop_cols = drop_cols.drop([i for i in predictors if i in drop_cols.index])
        drop_cols = drop_cols[drop_cols>corr_thresh].dropna(how='all').index
        drop_cols = [i for i in drop_cols if i in self._test_x.columns]
        self._test_x = self._test_x.drop(drop_cols, axis=1)
        if self._test_x.drop(list(predictors), axis=1).shape[1]==0:
            print('No features left to fit.')
            return 0
        return 1
