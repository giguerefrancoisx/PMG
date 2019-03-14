# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 09:27:54 2018

@author: tangk
"""

def get_units(channel_name):
    if 'Tmin' in channel_name or 'Tmax' in channel_name:
        return 'Time [s]'
    if 'FO' in channel_name:
        return 'Force [N]'
    elif 'AC' in channel_name:
        return 'Acceleration [g]'
    elif 'Excursion' in channel_name:
        return 'Excursion [mm]'
    elif '3ms' in channel_name:
        return 'Acceleration [g]'
    elif 'DS' in channel_name:
        return 'Deflection [mm]'
    elif '_x' in channel_name: 
        return 'Displacement [mm]'
    elif '_y' in channel_name:
        return 'Displacement [mm]'
    elif 'MO' in channel_name:
        return 'Moment [Nm]'
    elif 'Angle' in channel_name:
        return 'Angle [deg]'
    else:
        return channel_name
    
def rename(ch,names={}):
    """renames channel using names
    names is a dict of {names: replacement names}"""
    if ch in names:
        return names[ch]
    else:
        return ch

#Rename channels to more legible names
def renameISO(name):
    
    if 'HICR0000' in name:
        return 'HIC'
    elif 'HICR0036' in name:
        return 'HIC36'
    elif 'HICR0015' in name:
        return 'HIC15'
    elif 'BRIC' in name:
        return 'BRIC'
    elif 'NIJC' in name:
        return 'NIJ ' + name[8:10] 
    elif 'CNIJ' in name:
        return 'CNIJ ' + name[8:10]
    elif 'HEAD003S' in name:
        return 'Head 3ms clip'
    elif 'CHST003S' in name:
        return 'Chest 3ms clip'
    elif 'TIIN' in name:
        return 'Tibia Index ' + name[6:8]
    elif 'SEBE0000B6FO' in name:
        return 'Lap belt load'
    elif 'SEBE0000B3FO' in name:
        return 'Shoulder belt load'
    
#    name1 = name[:2]
    name2 = name[2:6]
    name3 = name[6:8]
    name4 = name[8:10]
    name5 = name[10:12]
    name6 = name[12:14]
    name7 = name[14]
    
    if name2=='HEAD':
        name2 = 'Head '
    elif name2=='CHST':
        name2 = 'Chest '
    elif name2=='PELV':
        name2 = 'Pelvis '
    elif name2=='SPIN':
        name2 = 'T'
    elif name2=='NECK':
        name2 = 'Neck '
    elif name2=='FEMR':
        name2 = 'Femur '
    elif name2=='CLAV':
        name2 = 'Clavicle '
    elif name2=='CVEH':
        name2 = 'Vehicle'
    elif name2=='ABDO':
        name2 = 'Abdomen '
    elif name2=='THSP':
        name2 = 'Chest '
    elif name2=='LUSP':
        name2 = 'Lumbar '
    elif name2=='ILAC':
        name2 = 'Iliac '
    elif name2=='ACTB':
        name2 = 'Acetabulum '
    elif name2=='KNSL':
        name2 = 'Knee '
    elif name2=='TIBI':
        name2 = 'Tibia '
    elif name2=='ANKL':
        name2 = 'Ankle '
    elif name2=='FOOT':
        name2 = 'Foot '
    elif name2=='SIME':
        name2 = 'Vehicle ' 
        
    if name3=='00':
        name3 = ''
    elif name3=='LE':
        name3 = 'Left '
    elif name3=='RI':
        name3 = 'Right '
    elif name3.isnumeric():
        name3 = name3
    elif name3=='LO':
        name3 = 'Lower '
    elif name3=='UP':
        name3 = 'Upper '
    elif name3=='CG':
        name3 = 'CG '
    elif name3=='TP':
        name3 = 'Top '
    
    if name4=='00':
        name4 = ''
    elif name4=='UP':
        name4 = 'Upper '
    elif name4=='LO':
        name4 = 'Lower '
    elif name4=='IN':
        name4 = 'Inner '
    elif name4=='OU':
        name4 = 'Outer '
    elif name4=='MI':
        name4 = 'Mid '
        
    name5 = ''
    
    if name6=='AC':
        name6 = 'Ac'
    elif name6=='FO':
        name6 = 'F'
    elif name6=='MO':
        name6 = 'M'
    elif name6=='AN':
        name6 = 'Angle '
    elif name6=='AV':
        name6 = 'Angular Velocity '
    elif name6=='DS':
        name6 = 'D'
    elif name6=='DC':
        name6 = 'D'
    
    if name7=='X':
        name7 = 'x'
    elif name7=='Y':
        name7 = 'y'
    elif name7=='Z':
        name7 = 'z'
    elif name7=='R':
        name7 = 'R'
    elif name7=='0':
        name7 = ''
    
    return name2 + name3 + name4 + name5 + name6 + name7

def rename_list(names):
    new = []
    for n in names:
        new.append(renameISO(n))
    return new