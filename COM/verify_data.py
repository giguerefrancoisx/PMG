# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 15:11:33 2019

Use this for checking that the test matrix is correct

@author: tangk
"""
import pandas as pd
import numpy as np
import xlwings as xw
from PMG.COM.writebook_xls2 import read_table_xw

def get_test_info():
    """gets test info from repertoire collision located in P:\Data Analysis\Tests"""
    directory = 'P:\\Data Analysis\\Tests\\'
    path = 'RepCollision.xls'
    
    try:
        rep_table = pd.read_excel(directory + path, header=None)
    except:
        rep_table = read_table_xw(directory + path)
    
    rep_table.columns = rep_table.loc[0]
    rep_table = rep_table.drop(0)
    return rep_table
