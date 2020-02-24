# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 16:04:56 2018
Read HDF5 written by h5py using h5py
@author: tangk

"""
import os
import pandas
import numpy
import re
import h5py
import xlwings as xw
from xlrd import XLRDError



def match_columns(table, regex='\d{2}[A-Z0-9]{14}'):
    """table is a pandas DataFrame
    Goes through row by row until something resembling channel names is found
    Optionally, takes a custom regex. The default one looks for the default 
    coding standard for channel names. rows can either be the first row to
    start searching or a range of rows to search over"""
    
    for i in table.index:
        contains_exp = table.loc[i].str.contains(regex)
        if contains_exp.any():
            return i
    return -1


def get_colnames(table, channel_exp='\d{2}[A-Z0-9]{14}', time_exp='000000000000TI00'):
#def get_colnames(table, channel_exp='\d{2}[A-Z0-9]{14}', time_exp='T_10000_0'):
    row = match_columns(table, regex=channel_exp)
    t_row = match_columns(table, regex=time_exp)
    if row!=t_row: 
        raise Exception('Error: time column and channel column were not in the same row. Check data.')
    names = table.loc[row].to_dict()
    return names, row

    
def get_data(table, row_start=0, colnames={}):
    """table is a pandas DataFrame
    Returns the row and column where the data begin.
    Assumes that single value measurements e.g. hic have values 0 after index 0"""
    if len(colnames)>0:
        table = table.rename(colnames, axis=1)
    
    is_str = table.applymap(type)==str
    table[is_str] = numpy.nan
    table = table.dropna(axis=1, how='all').dropna(axis=0, how='any')
    return table


def read_table_xw(path, empty_val='NA'):
    """reads a table using xlwings"""
    book = xw.Book(path)
    table = pandas.DataFrame(book.sheets[0].used_range.options(empty=empty_val).value)
    book.close()
    return table


def delete_columns(cols_to_delete, df, row_index):
    """deletes col_to_delete from df
    col_to_delete (list of strings): names of column to delete
    df (dataframe): dataframe of data
    row_index (int): index of the row where channel names are stored"""
    drop = []
    for col in cols_to_delete:
        tcol = df.loc[row_index][df.loc[row_index]==col]
        drop.append(tcol.index)
    if len(drop)>0:
        df = df.drop(drop, axis=1)
    return df


def read_excel(path, delete_cols=[]):
    """reads an excel file (extension .xls or .xlsx) using pandas or xlwings."""
    try:
        testframe = pandas.read_excel(path,sheet_name=None, header=0,index_col=0, skiprows=[1,2],dtype=numpy.float64)
        testframe = pandas.concat(list(testframe.values()), axis=1)
    except:
        testframe = read_table_xw(path)
        colnames, row_index = get_colnames(testframe)
        if 'T_10000_0' not in colnames.values():
            print('Warning: T_10000_0 not in ' + path)
        testframe = testframe.rename(colnames, axis=1)
        testframe = get_data(testframe)
    if len(delete_cols)>0:
        testframe = testframe.drop([col for col in delete_cols if col in testframe.columns], axis=1)
    return testframe

# TO DO: parse vehicle info
def import_faro_data(path):
    """imports and preprocesses faro data specified by path"""
    faro_data = read_faro(path)
    split = split_sections(faro_data)
    split['crash_info'] = parse_crash_info(split['crash_info'])
    split['dummy_info'] = parse_dummy_info(split['dummy_info'])
    split['faro_points'] = parse_faro_points(split['faro_points'], 
                                             {'Cible': split['crash_info']['TC1'],
                                              'Bélier': split['crash_info']['TC2']},
                                              treat_exceptions='pad')
    return split


def read_faro(path, delete_cols=[]):
    """reads the excel file where the faro measures are stored
    returns faro_data, an array of all the elements with blank spaces dropped"""
#    faro_data = pandas.read_excel('C:\\Users\\tangk\\Desktop\\TC18-214VSTC13-024Mesures Faro.xls', header=None, na_values=[''], keep_default_na=False)
    faro_data = pandas.read_excel(path, header=None, na_values=[''], keep_default_na=False)
    faro_data = faro_data.reset_index(drop=True).dropna(axis=0, how='all').dropna(axis=1, how='all')
    faro_data = numpy.concatenate(faro_data.apply(lambda x: tuple(x.dropna()), axis=1).values)
    return faro_data

def split_sections(faro_data):
    """splits the sections of faro_data into: crash_info, dummy_info, vehicle_info, faro_points"""    
    # crash_info: starts at the beginning of the doc and continues until just before 'Cible'
    # last entry is 'Vitesse'
    cib_loc = numpy.where(faro_data=='Cible')[0]
    crash_info_end = cib_loc[0]-1
#    crash_info_end = crash_info_end[crash_info_end > numpy.where(faro_data=='Vitesse')[0][0]]
#    crash_info_end = crash_info_end[numpy.where(faro_data[crash_info_end+1] == '11')[0][0]]
    
    # dummy_info: starts at 'Cible' and ends where 'Cible' follows 'Attitudes (mm)'
    dummy_info_end = numpy.where(faro_data=='Attitudes (mm)')[0][0]

    # vehicle info: ends with 'Mesures'
    vehicle_info_end = numpy.where(faro_data=='Mesures Véhicule')[0][0]
    
    split = {'crash_info': faro_data[:crash_info_end],
             'dummy_info': faro_data[crash_info_end:dummy_info_end],
             'vehicle_info': faro_data[dummy_info_end:vehicle_info_end],
             'faro_points': faro_data[vehicle_info_end:]}
    return split


def parse_crash_info(x):
    """takes an input of faro data of type crash_info and parses """
    
    crash_info = {'TC1': x[0],
                  'TC2': x[1],
                  'test_type': x[2]}
    start = numpy.where(x=='TC_Vehicule')[0][0]
    x = x[start:]
    # assume that each element in x follows keyword --> value 
    if len(x)%2>0: # if an odd number of elements, assume vitesse has not been filled
        x = x[:-1].reshape(-1, 2).tolist()
    else:
        x = x.reshape(-1, 2).tolist()
    
    crash_info_df = pandas.DataFrame(columns=range(2))
    i = 0
    while len(x)>0:
        kwarg = x.pop(0)
        crash_info_df.at[kwarg[0], i] = kwarg[1]
        i = 1 - i 
    crash_info['data'] = crash_info_df
    return crash_info


def parse_dummy_info(x):
    """takes an input of fara data of type dummy_info and parses"""
    dummy_info = {'Pos': [], 'Dummy': [], 'Dummy_id': [], 'Params': []}
    
    r = re.compile('^\d{2}$')
    pos = list(filter(r.match, x.tolist()))
    starts = list(filter(lambda i: x[i] in pos, range(len(x))))
    ends = starts[1:] + [len(x)-1]
    
    for start, end in zip(starts, ends):
        # assume the data always starts with: 
        # [0] position
        # [1] dummy
        # [2] dummy id
        dummy_info['Pos'].append(x[start])
        dummy_info['Dummy'].append(x[start+1])
        dummy_info['Dummy_id'].append(x[start+2])
        kws = ['Facing', 'Child restraint', 'Child seat restraint',
               'Tether', 'Seat type', 'Seat config', 'Seat base', 'Belt side',
               'Seat belt adj', 'Load cell', 'Handle', 'Contact', 'Bélier']
        if end-start-2>0:
            params = [i for i in x[start+3:end] if i not in kws]
        else:
            params = ['none']
        dummy_info['Params'].append(params)
    dummy_info = pandas.DataFrame(dummy_info)
    return dummy_info
    
# work on this later
def parse_faro_points(x, test_info, treat_exceptions=None, verbose=0):
    """takes an input of faro data of type faro_points and parses the data. Returns
    a dict with the details of each subsection.
    
    test_info is a dict of {'Cible': TC id of cible, 'Bélier': TC id of belier}
    
    treat_exceptions specifies how to treat cases where the number of elements is not 
    divisible by the number of columns. options are:
        None: will return an exception if the 
        'remove_last': will remove the remainder so that the array can be reshaped
        'pad': will pad with 'NA' until the array can be reshaped"""

    # get all the headers
    all_headers = ['Mesures Véhicule', 'Offset du point 1060', 'Mesures AX', 
                   'Mesures BX', 'Mesures DPD','Emplacement des accéléromètres'] +\
                       ['Mannequin {0}{1}'.format(i,j) \
                        for i in ['11','12','13','14','15','16','17','18','19',
                                  '21','22','23','24','25','26','27','28','29'] \
                                  for j in ['',' Texte',' Remarques'] ]
    
    faro_points = dict.fromkeys(all_headers)
    
    # start and end indices for each section as specified in the headers
    starts = numpy.array(list(filter(lambda i: x[i] in all_headers, range(len(x))))) 
    ends = numpy.append(starts[1:], len(x))
    
    for start, end in zip(starts, ends):
        # get the header and column names
        header_name = x[start]
        if 'Texte' in header_name:
            if 'Cible' in x[start:end]:
                columns = [test_info['Cible'], 'Description']
            elif 'Bélier' in x[start:end]:
                columns = [test_info['Bélier'], 'Description']
        elif 'Remarques' in header_name:
            columns = ['Remarques']
        else:
            columns = ['ID','X','Y','Z','Description']
        
        # find where values start
        column_indices = [i for i in range(start,end) if x[i] in columns]
        if (len(column_indices)==0):
            if verbose:
                print('No data in section ' + header_name)
            continue
        elif (end - column_indices[-1] - 1 < len(columns)):
            if verbose:
                print('No data in section ' + header_name)
            continue
        
        measures = x[column_indices[-1]+1:end]
        remainder = len(measures)%len(columns)
        if remainder!= 0:
            if treat_exceptions is None:
                raise Exception('Error parsing the data for ' + header_name)
            elif treat_exceptions=='remove_last':
                measures = measures[:-remainder]
            elif treat_exceptions=='pad':
                measures = numpy.append(measures, numpy.repeat('NA', len(columns)-remainder))
                
        faro_points[header_name] = pandas.DataFrame(measures.reshape(-1,len(columns)), columns=columns)
    return faro_points

    

def delete_tests(testlist, directory):
    """deletes tests specified in testlist.
    testlist is a list of the tests to be deleted from the HDF5 store.
    directory specifies where the tests are stored"""
    status = check_filenames(testlist)
    if status=='ok':
        ntest = len(testlist)
        with h5py.File(directory + 'Tests.h5') as test_store:
            for i, test in enumerate(testlist):
                del test_store[test.replace('-','N')]
                print('Deleted {0}/{1}'.format(i+1, ntest))
    else:
        print('Could not delete tests. Check file names')
        return         


def check_testframe(tf):
    """checks testframe to make sure the data imported and column names 
    are OK. Returns 0 if autocheck failed and 1 if autocheck passed"""
    if isinstance(tf, pandas.core.series.Series):
        return 'series'
    if min(tf.shape)==0 or tf.isna().all().all():
        return 'empty'
    if len(tf)<2:
        return 'truncated'
    if not tf.applymap(numpy.isreal).all().all():
        return 'nonreal'
    if 'T_10000_0' in tf.columns:
        tf = tf.drop('T_10000_0', axis=1)
    ch_names = tf.filter(regex='[A-Z0-9]\d[A-Z0-9]{14}', axis=1).columns
    if len(ch_names) !=  len(tf.columns):
        return 'unrecognized columns: ' + str([i for i in tf.columns if i not in ch_names])
    return 'ok'


def check_filenames(filenames, regex='[TS][CE]\d{2}-\d{3,4}.*'):
    """checks file names before reading files to make sure
    the files to read are OK. Filenames is a list-like. Returns 'ok' if 
    autocheck passed and 'unmatched names' otherwise"""
    rc = re.compile(regex)
    matched = list(filter(rc.match, filenames))
    if len(matched) != len(filenames): 
        return 'unmatched names'
    else:
        return 'ok'


# to do: add option of editing testframe and re-checking
def writeHDF5(directory, file_check=1, data_check=1, delete_cols=[]):
    """reads .xls, .xlsx, and .csv data files and writes them to HDF5
    Optionally check filenames and data before reading and writing. Values of
    file_check and data_check correspond to:
        0: no check
        1: exit if autocheck fails
        2: if autocheck passes, print and continue with user input"""

    allfiles = [file for file in os.listdir(directory) if file.endswith(('.xls','.xlsx','.csv'))]
    
    # check file names
    if file_check>0:
        status = check_filenames(allfiles)
        if status!='ok':
            print(status, 'Files found:', allfiles, sep='\n')
            return
        elif file_check>1:
            print(status, 'Files found:', allfiles, sep='\n')
            if input('continue? [y/n]')=='n':
                return
    
    print('Reading files:')
    count = len(allfiles)
    i = 1
    
    with h5py.File(directory+'Tests.h5') as test_store:
        stored_tests = list(test_store.keys())
    
    for filename in allfiles:
        per = i/count*100
        print('\n') #clear the screen hack. if cmd, use os.system('cls')
        print('{:.0f} % Complete'.format(per))
        l = ['|']*i+[' ']*(count-i)+['| {:>2}/{}'.format(i, count)]
        print(''.join(l))
        
        new_name = re.search('\w{4}-\d{3,4}(_\d+)?',filename).group().replace('-','N')
        if new_name in stored_tests:
            print(new_name + ' already in keys! skipping...')
            i = i + 1
            continue
        if filename.endswith(('.xls','.xlsx')):
            testframe = read_excel(directory+filename, delete_cols=delete_cols)
        elif filename.endswith(('.csv')) and ('channel_names' not in filename) and ('test_names' not in filename):
            testframe = pandas.read_csv(directory+filename, dtype=numpy.float64)
        
        if data_check>0:
            status = check_testframe(testframe)
            if status!='ok':
                print(new_name,status,'df size:', testframe.shape,'df columns:',testframe.columns, sep='\n')
                return
                    
            if data_check>1:
                print(new_name,status,'df size:', testframe.shape,'df columns:',testframe.columns, sep='\n')
                if input('continue? [y/n]')=='n':
                    return
        testframe.columns = ['X' + i for i in testframe.columns]
        types = [(i, numpy.float64) for i in testframe.columns]
        
        
        with h5py.File(directory+'Tests.h5') as test_store:
            ds = test_store.create_dataset(new_name,shape=(testframe.shape[0],),dtype=types)
            ds[...] = testframe.apply(tuple,axis=1).values
            
        i = i+1

def write_angle(directory):
    allfiles = os.listdir(directory)
    print('Reading files:')
    for filename in allfiles:
        if filename.endswith(('.xls','.xlsx')):
            book = xw.Book(directory+filename)
            testframe = pandas.DataFrame(book.sheets[0].range('A4').expand().value,
                                         columns=['Time','Up_x','Up_y','Down_x','Down_y'])
            book.close()
        else:
            continue
            
        new_name = re.search('\w{4}-\d{3,4}(_\d+)?',filename).group().replace('-','N')
        types = [(i, numpy.float64) for i in testframe.columns]
        with h5py.File(directory+'Tests.h5') as test_store:
            if new_name in list(test_store.keys()):
                print(new_name+' already in keys! skipping write...')
                continue
            ds = test_store.create_dataset(new_name,shape=(testframe.shape[0],),dtype=types)
            ds[...] = testframe.apply(tuple,axis=1).values

        

        
#%%%
if __name__=='__main__':
    from PMG.read_data import update_test_info
    
    def write_tc():
        directory_tc = 'P:\\Data Analysis\\Data\\TC\\'
        delete_cols = ['T']
        writeHDF5(directory_tc, file_check=1, data_check=1, delete_cols=delete_cols)
        update_test_info()
    
    def write_se():
        directory_se = 'P:\\Data Analysis\\Data\\SE\\'
        delete_cols = ['Time Upper', 'Upper Bond', 'Time Lower', 'Lower Bond', 'T']
        writeHDF5(directory_se, file_check=1, data_check=1, delete_cols=delete_cols)
        update_test_info()
    
    def write_angle_213():
        directory_angle = 'P:\\Data Analysis\\Data\\angle_213\\'
        write_angle(directory_angle)
    
    write_tc()
#    write_angle_213()
#    write_se()