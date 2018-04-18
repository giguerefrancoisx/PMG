# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 13:03:35 2017

@author: giguerf
"""
import pandas

def openHDF5(directory, channels=None):
    """
    Reads all channel workbooks from the current directory into a single
    dictionary. Also creates a 'time' series

    Input:
    ----------
    directory : str
        path to data storage.
    channels : str, default None
        optional, return data for a small list of channels from the set

    Returns:
    ----------
    time :
        series containing time channel.
    fulldata :
        dictionary containing a DataFrame for each channel.
    """
    try:
        with pandas.HDFStore(directory+'Channels.h5', mode='r+') as data_store:
            fulldata = {}

            try:
                if channels is not None:
                    if isinstance(channels, str):
                        channels = [channels]
                    for channel in channels:
                        chdata = data_store.select('C'+channel)
                        fulldata[channel] = chdata.iloc[:,1:]

                else:
                    for key in data_store.keys():
                        chdata = data_store.select(key)
                        fulldata[key[2:]] = chdata.iloc[:,1:]
            except KeyError:
                raise KeyError('Channel {} has not been written to the '
                               'channels HDF5 store yet. You must first use '
                               'the writeHDF5 function'.format(channel))

            time = chdata.iloc[:4100,0].round(4)
    except OSError:
        raise OSError('File Not Found. You are not conected to P:/Projects '
                      'or you must first create the channels HDF5 store using '
                      'the writeHDF5 function\n')

    return time, fulldata

#import tables as tb
#tb.file._open_files.close_all()    #Close all open stores