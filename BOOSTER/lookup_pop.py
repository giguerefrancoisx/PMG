# -*- coding: utf-8 -*-
"""
LOOKUP POPULATIONS
    Using a lookup table, separates the input list of TCNs into populations: the group 
    specified and all others.
    
Created on Thu Oct 19 11:11:01 2017

@author: giguerf
"""
import pandas

def lookup_pop(TCNs, group):

    population = pandas.DataFrame(TCNs,columns = ['ALL'])
    table = pandas.read_excel('C:/Users/gigue/Desktop/BOOSTER/boostertable.xlsx', index_col = 0)
    
    population['Type'] = table.loc[population['ALL'],'Type'].tolist()
    population['Vitesse'] = table.loc[population['ALL'],'Vitesse'].tolist()
    population['Marque'] = table.loc[population['ALL'],'Marque'].tolist()
    population['Modele'] = table.loc[population['ALL'],'Modele'].tolist()
    population['Group'] = table.loc[population['ALL'],'Group'].tolist()

    subset = population[population['Group'] == group]
    subset = subset.drop_duplicates()
    population = population.drop(subset.index)
    population = population.dropna(axis = 0, thresh = 2)

    return population, subset