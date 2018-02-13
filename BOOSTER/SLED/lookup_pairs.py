# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 13:30:48 2017

@author: giguerf
"""
import pandas

# make sure that there are no duplicate entries in lookup table
def lookup_pairs(TCNs):

        #lookup with OLD as index
        try:
            pairs = pandas.DataFrame(TCNs,columns = ['OLD'])
            table = pandas.read_excel('P:/SLED/sledtable.xlsx', index_col = 0)
            pairs['NEW'] = table.loc[pairs['OLD'],'NEW'].tolist()
            pairs = pairs[pairs.NEW.notnull()] #pairs.dropna() ?
            pairs['SEAT'] = table.loc[pairs['OLD'],'SEAT'].tolist()
            pairs['GROUP'] = table.loc[pairs['OLD'],'GROUP'].tolist()
        except:
            print('Except')
            pass
        #reverse, using NEW as index
        try:
            pairs2 = pandas.DataFrame(TCNs,columns = ['NEW'])
            table2 = pandas.read_excel('P:/SLED/sledtable.xlsx', index_col = 5)
            pairs2['OLD'] = table2.loc[pairs2['NEW'],'OLD'].tolist()
            pairs2 = pairs2[pairs2.OLD.notnull()]
            pairs2['SEAT'] = table2.loc[pairs2['NEW'],'SEAT'].tolist()
            pairs2['GROUP'] = table.loc[pairs2['OLD'],'GROUP'].tolist()
        except:
            print('Except')
            pass
        #merge results for full list
        pairs = pairs.merge(pairs2, how = 'outer')
        
        return pairs