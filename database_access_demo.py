# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 16:37:15 2020

@author: giguerf
"""

# install mysql-connector
import mysql.connector
import pandas as pd
from contextlib import contextmanager

@contextmanager
def open_connection(config):
    try:
        db = mysql.connector.connect(**config)
        if db.is_connected():
            print('Connected to MySQL Server version ', db.get_server_info())
        yield db
    except Exception as err:
        print('Warning:\n',err)
    finally:
        if db.is_connected():
            db.close()
            print('Now disconnected')

@contextmanager
def use_cursor(db, **kwargs): #use dictionary=True for rows as dict
    try:
        curs = db.cursor(**kwargs)
        print('Cursor active')
        yield curs
    except Exception as err:
        print('Warning:\n',err)
    finally:
        curs.close()
#        print('Cursor reset')

with open_connection(CONNECTION_CONFIG) as mydb:
    with use_cursor(mydb) as faro_data:
#    faro_data = mydb.cursor() #dictionary=True for rows as dict
#        print('cursor activated')
#        TC = 'TC20-115'
#        faro_data.execute(f'SELECT * FROM faro_data.texte where TC_Vehicule like "{TC}"')
        faro_data.execute('SELECT * FROM faro_data.texte')
        count = faro_data.rowcount
        first_five = faro_data.fetchmany(size=5)
        columns_found = faro_data.column_names
        new_count = faro_data.rowcount
        the_rest = faro_data.fetchall()
        newest_count = faro_data.rowcount #gives cumulative count of records fetched
        df = pd.DataFrame.from_records(the_rest, columns = columns_found)

#%% Non-Context Manager version:
#try:
#    mydb = mysql.connector.connect(**CONNECTION_CONFIG)
#    if mydb.is_connected():
#        db_Info = mydb.get_server_info()
#        print("Connected to MySQL Server version ", db_Info)
#        faro_data = mydb.cursor() #dictionary=True for rows as dict
##        faro_data.execute('SELECT * FROM faro_data.texte where TC_Vehicule like %s', ('TC20-115',))
#        TC = 'TC20-115'
#        faro_data.execute(f'SELECT * FROM faro_data.texte where TC_Vehicule like "{TC}"')
#        count = faro_data.rowcount
#        first_five = faro_data.fetchmany(size=5)
#        columns_found = faro_data.column_names
#        new_count = faro_data.rowcount
#        the_rest = faro_data.fetchall()
#        newest_count = faro_data.rowcount
#        df = pd.DataFrame.from_records(the_rest, columns = columns_found)
#        #remember to close and reset cursor:
#        faro_data.close()
#except Exception as err:
#    print(err)
#finally:
#    if mydb.is_connected():
#        print('Now disconnecting...')
#        mydb.close()