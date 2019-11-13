# -*- coding: utf-8 -*-
"""
Classes and functions for accessing MME format data

@author: Francois Giguere
"""
from pathlib import Path
import numpy as np
import pandas as pd # Beware time-consuming pandas import...

class MMEData():
    '''
    The class that handles test data and parameters for MME format

    List of attributes and methods:
        self.directory : Path
            root directory
        self.file : Path
            path to .mme file
        self.info : dict
            dictionary of parameters in .mme file
        self.test_objects : list
            list of test object codes present in the data
        self.channels : list
            list of channels as channel objects
        self.channels_as_str : list
            list of channels as strings
        self.to_dataframe()
            output the test data as pandas DataFrame
        self.channels_from_list()
            returns the Channel objects matching the list of strings provided
        self.get_channel_parts()
            return channel list grouped by iso parts as a pandas DataFrame

    Input:
    --------
    path : str or pathlib.Path
        Path to the mme data. This can point to the folder containing the .mme file,
        the .mme file itself, or the channels folder.

    Returns:
    ---------
    MMEData instance
        The object containing the test information and data.

    Examples:
    ---------
    >>> my_test = MMEData('C:/path/to/test_name.mme')
    Directory: C:/path/to/
    Test: test_name

    >>> my_test.channels
    [10CVEHCG0000ACXP, 10CVEHCG0000ACYP, 10CVEHCG0000ACZP, ...

    This returns channel objects, see Channel() documentation. To get iso codes
    as strings:

    >>> my_test.channels_as_str
    ['10CVEHCG0000ACXP', '10CVEHCG0000ACYP', '10CVEHCG0000ACZP', ...


    >>> my_test.to_dataframe()
          10CVEHCG0000ACXP        ...         11ABDOLESUERFOYB
    0            -0.022442        ...                 0.021263
    1            -0.006967        ...                 0.001306
    ...                ...        ...                      ...
    '''

    def __init__(self, path):
        if isinstance(path, Path):
            self.path = path
        else:
            self.path = Path(path)
        self._validate_path()

        self._load_params()
        self._load_channels()

    def _validate_path(self):
        if not self.path.is_file(): # A folder was given, assume it's the root folder
            self.directory = self.path
        elif self.path.suffix.lower() == '.mme':
            self.directory = self.path.parent
        else:
            raise ValueError('Path must be .mme file or a folder containing one!')

        # Find mme file(s)
        files = list(self.directory.glob('**/*.mme'))
        if len(files)>1:
            raise ValueError('Multiple .mme files found in this directory!')
        elif len(files) == 0:
            raise ValueError('No .mme file found in this directory!')
        else:
            self.file = files[0]

    def _load_params(self):
        self.info = {}
        for line in self.file.read_text().split('\n'):
            if line == '':
                continue
            else:
                param_name, value = line.split(':', maxsplit=1) # Maxsplit of 1 to avoid splitting datetimes, etc.
                self.info[param_name.rstrip()] = value

    def _load_channels(self):
        chn_file = list((self.directory/'channel').glob('**/*.chn'))[0]
        chn_list = {}
        for line in chn_file.read_text().split('\n'):
            if line == '':
                continue
            param_name, value = line.split(':', maxsplit=1)
            if param_name.startswith('Name'):
                chn_list[param_name[16:19]] = value[:16]
        self.channels = []
        for file in sorted((self.directory/'channel').iterdir()):
            if file.suffix != '.chn':
                code = chn_list[file.suffix[1:]]
                self.channels.append(Channel(file, code))
        self.channels_as_str = [chn.code for chn in self.channels]
        self._lookup = dict(zip(self.channels_as_str,self.channels))
        self.test_objects = sorted(set(ch[0] for ch in chn_list.values())) # strings representing object codes. is there a more efficient way?
        # Make the above property a function describing each object? or get rid of it entirely?

    def __repr__(self):
        return f"MMEData('{self.file.as_posix()}')"

    def to_dataframe(self): # Will using np arrays for Channels speed things up?
        '''Returns test data as pandas DataFrame, with channels as columns.
        Currently supports one time base (taken from the first channel),
        although MME files can have a time base for each channel.'''
        data = {'Time':self.channels[0].time[:4100]}
        for code, channel in self._lookup.items():
            data[code] = channel.data[:4100] #TODO: fix uneven length channel issue
        return pd.DataFrame.from_dict(data)

    def channels_from_list(self, chlist):
        '''Given a list of iso codes, returns the Channel objects associated.
        Returns None for channels that do not exist in the set.'''
        return [self._lookup.get(ch,None) for ch in chlist]

    def get_channel_parts(self): # @property?
        '''Splits each ISO code into it's parts using Channel.to_iso_parts()
        Returns a pd.DataFrame with the parts as columns and channels as rows'''
        return pd.DataFrame(data=[ch.to_iso_parts() for ch in self.channels],
                            columns=['VEH','POS','MAIN','FL1','FL2','FL3','PD','D','FC'])

class Channel():
    '''
    The class that handles channel data and parameters for MME format

    List of attributes and methods:
        self.number : str
            channel number (extension of channel file)
        self.code : str
            channel's ISO code
        self.name : str
            channel's plaintext name
        self.info : dict
            dictionary of parameters in channel file
        self.data : list
            list of data points in channel file
        self.time : list
            time base for the channel, in seconds
        self.time_ms : list
            time base for the channel, in milliseconds
        self.to_iso_parts()
            split the ISO code into it's parts.

    Input:
    --------
    path : str or pathlib.Path
        Path to the channel data. This must point to the channel file itself
        ex. test_name/channel/test_name.001

    Returns:
    ---------
    Channel instance
        The object containing the channel information and data.

    Examples:
    ---------
    >>> my_test = MMEData('C:/path/to/test_name.mme')

    >>> my_channel = my_test.channels[0]

    >>> my_channel.code
    '10CVEHCG0000ACXP'

    >>> my_channel.data
    [-0.02244193343, -0.006967360652, 0.03195831689, ...
    '''
    def __init__(self, path, code):
        self.path = path
        self.number = self.path.suffix[1:]
        self.code = code
        self._info = None
        self._name = None
        self._data = None
        self._time = None
        self._time_ms = None

    def _load(self):
        self._info = {}
        self._data = []
        with open(self.path.as_posix(), 'r') as file:
            for line in file:
                if line == '':
                    continue
                if ':' in line:
                    param_name, value = line.split(':', maxsplit=1)
                    self._info[param_name.rstrip()] = value.rstrip()
                else:
                    self._data.append(float(line))
        self._data = np.array(self._data)
        try:
            assert self.code == self._info['Channel code']
        except AssertionError:
            raise AssertionError(f'Channel code for file {self.path.name} does not match code given in .chn file!')
        try:
            self._name = self._info['Name of the channel']
        except KeyError:
            raise KeyError(f'Unable to load channel name. "Name of channel" is missing')
        # TODO: handle 'Laboratory channel code', and 'Customer channel code'

        try:
            dt = float(self._info['Sampling interval'])
            t0 = float(self._info['Time of first sample'])
            n_samples = int(self._info['Number of samples'])
        except ValueError:
            raise ValueError(f'Timebase cannot be created for channel: {self._code}\nInvalid value for one of:\n"Sampling interval", "Time of first sample", or "Number of samples"')
        self._time = np.linspace(t0, t0+(n_samples-1)*dt, n_samples)
        self._time_ms = self._time*1000 # Now that this is a np array, storing this probably isn't necessary since its easy to calculate

    def _get_attr(self, attr):
        if getattr(self, attr) is None:
            self._load()
        return getattr(self, attr)

    @property
    def info(self):
        return self._get_attr('_info')

    @property
    def name(self):
        return self._get_attr('_name')

    @property
    def data(self):
        return self._get_attr('_data')

    @property
    def time(self):
        return self._get_attr('_time')

    @property
    def time_ms(self):
        return self._get_attr('_time_ms')

    def to_iso_parts(self):
        '''
        ISO codes are comprised of 16 digits:

        * 'D'        Test object number
        * 'D'        Sensor position
        * 'LLLL'     Main location
        * 'LL'       Fine location 1
        * 'LL'       Fine location 2
        * 'LL'       Fine location 3
        * 'LL'       Physical Dimension
        * 'L'        Principal Direction
        * 'L'        Filter class

        Where D is primarily a digit and L is primarily a letter
        See ISO 13499 documentation for valid codes.

        This method separates the code into is components:

        * VEH, POS, MAIN
        * FL1, FL2, FL3
        * PD, D, FC

        Returns:
        ---------
        parts: list
            List of separated strings
        '''
        parts = []
        for s, e in zip([0,1,2,6,8,10,12,14,15],[1,2,6,8,10,12,14,15,16]):
            parts.append(self.code[slice(s,e)])
        return parts

    def __repr__(self):
        return self.code

if __name__ == '__main__':
    my_path = 'P:/2020/20-6000/20-6020 (ARRIÈRE BUS-BUS)/TC06-002 NewFlyer BUS (Bullet) (TC06-001 Target)/PMG/TC06-002/TC06-002.mme'
    test = MMEData(my_path)
    print(test)
#    test.channels[0].data
#    df = test.to_dataframe()
