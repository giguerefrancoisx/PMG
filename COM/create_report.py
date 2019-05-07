# -*- coding: utf-8 -*-
"""
Created on Wed May  1 12:46:09 2019
Create reports as follows. 

FEATURES: 
For each test, create plots depending on the type of test.
If the test is a one-sample test of the mean, plot:
    1. The distribution of the samples
If the test is a two-sample test of the mean, plot:
    1. Bar plots of the mean and standard deviation if unpaired. Sideways plots of the difference if paired.
If the test is for the significance of explanatory variables, plot:
    1. Relational plots between each feature and the independent variable(s). If one or more of the variables is 
    discrete, do the plot in a different colour for every combination of the discrete variables. 
If the test is for the significance of explanatory models, plot:
    1. Relational plots between each feature and the independent variable(s). If one or more of the variables is 
    discrete, do the plot in a different colour for every combination of the discrete variables.  

TIME SERIES
Figure this out later
@author: tangk
"""
import seaborn as sns
import matplotlib.pyplot as plt
from PMG.read_data import PMGDataset
from PMG.COM.plotfuns import *

directory = 'P:\\Data Analysis\\Projects\\Y7 Pulse Study\\'


def get_data(directory):
    dataset = PMGDataset(directory)
    dataset.get_data(['stats','features'])
    return dataset

def get_testtype(test_params):
    model_dictionary = {'wilcox.test': 'mean_test',
                        't.test': 'mean_test',
                        'lm': 'variable_test',
                        'lmer': 'model_test'}
    testname = maybe_squeeze(test_params['testname'])
    if testname in ['variable_test','model_test']:
        return model_dictionary[testname]
    if 'test2' in test_params:
        return 'two_sample_test'
    else:
        return 'one_sample_test'
    
    
def get_plots(dataset):
    """
    iterates through each test to generate the plots
    """
    all_plots = {}
    features = dataset.features
    for test_params in dataset.stats.params['test']:
        test_name = maybe_squeeze(test_params['name'])
        testtype = get_testtype(test_params)
        if testtype=='one_sample_test':
            plots = plot_one_sample_features(features, test_params)
        elif testtype=='two_sample_test':
            plots = plot_two_sample_features(features, test_params)
        elif testtype=='variable_test':
            plots = plot_relational_features(features, test_params)
        elif testtype=='model_test':
            plots = plot_relational_features(features, test_params)
        all_plots[test_name] = plots    
    return all_plots

def plot_one_sample_features(features, test_params):
    # distribution of the samples
    return []


def maybe_squeeze(item):
    """if the item is a list of one element, get the element"""
    if isinstance(item, (list, tuple)) and len(item)==1:
        return item[0]
    else:
        return item
    

def plot_two_sample_features(features, test_params):
    # get numerical features only
    features = features.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64'])
    fig, axs = plt.subplots(nrows=features.shape[1])
    # plot bars
    plots = []
    
    test1_name = maybe_squeeze(test_params['test1_name'])
    test2_name = maybe_squeeze(test_params['test2_name'])
    test1 = test_params['test1']
    test2 = test_params['test2']
    for col in features.columns:
        x = features.loc[test1+test2, [col]]
        x['type'] = [test1_name]*len(test1) + [test2_name]*len(test2)
#        fig, ax = plt.subplots()
        ax = sns.barplot(x='type', y=col, data=x)
        ax.figure = None
        plots.append(ax)
        plt.close(fig)
    return plots

def plot_relational_features(features, test_params):
    # plot scatter or bar, depending on the type of data
    variables = test_params['variables']
    return []
            
            
if __name__=='__main__':
    dataset = get_data(directory)
    all_plots = get_plots(dataset)
    for key, plots in all_plots.items():
        fig = plt.figure(figsize=(6, 4*len(plots)))
        for plot in plots:
            plot.set_figure(fig)
            fig.add_subplot(plot)
            break












#from PMG.COM.arrange import *
#from PMG.COM.plotfuns import *
#from PMG.COM.helper import *
#
#
#
#data = initialize_report_data(directory, ['features','timeseries'])
#table = data['table']
#t = data['t']
#chdata = data['timeseries']
#features = data['features']
#
#
#report_info = [{'name': 'Chest 3ms clip comparison',
#                'section_type': 'plot',
#                'datatype': 'features',
#                'function': 'sns.barplot',
#                'args': [],
#                'kwargs': {'x': 'Model',
#                           'y': 'Head_3ms',
#                           'hue': 'Pulse',
#                           'data': table,
#                           'ax': None}},
#               {'name': 'Chest Acx comparison',
#                'section_type': 'plot',
#                'datatype': 'timeseries',
#                'function': 'plot_overlay',
#                'args': [t,
#                         arrange_by_group(table, chdata['12HEAD0000Y7ACXA'], 'Pulse')],
#                'kwargs': {}}]
#
#for section in report_info:
#    if section['section_type']=='plot':
#        fig, ax = plt.subplots()
#        if 'ax' not in section['kwargs']:
#            section['args'].insert(0, ax)
#        res = do_anything(section['function'], *section['args'], **section['kwargs'])
