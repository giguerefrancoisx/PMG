# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 11:58:07 2018

@author: tangk
"""
import pandas as pd
# reads data and returns dictionary 

# as a single dictionary
# filename is a list of strings containing file names within a directory
def read_merged(directory,filename):
    full_data = {}
    for file in filename:
        print('Reading file ' + directory + file + '(SAI).csv')
        full_data[file] = pd.read_csv(directory + file + '(SAI).csv')
    return full_data

# as two dictionaries: target and bullet
# tarfile is a list of strings containing file names of targets within a directory
# bulfile is a list of strings containing file names of bullets within a directory
def read_split(directory,tarfile,bulfile):
    tar = {}
    bul = {}
    for i in len(tarfile):
        tar[tarfile[i-1]] = pd.read_csv(directory + tarfile[i-1] + '(SAI).csv')
        bul[bulfile[i-1]] = pd.read_csv(directory + bulfile[i-1] + '(SAI).csv')
    return tar, bul # return target, bullet files respectively. 
