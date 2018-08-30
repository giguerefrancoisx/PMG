# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 09:27:54 2018

@author: tangk
"""

# only for sled 3ms clips
def inverse_rename(name,pos):
    if name==name.split(" "):
        name = name.split(".")
    else:
        name = name.split(" ")

    if name[0]=='Head':
        name1 = 'HEAD'
    elif name[0]=='Chest' and ('Ac' in name[1]) or (name[1]=='3ms'):
        name1 = 'CHST'
    elif name[0]=='Pelvis':
        name1 = 'PELV'
    elif name[0]=='Femur':
        name1 = 'FEMR'
    elif name[0]=='Acetabulum':
        name1 = 'ACTB'
    elif name[0]=='NIJ':
        name1 = 'NIJC'
    elif name[0]=='HIC15':
        name1=='HICR0015'
    elif name[0]=='BRIC':
        name1 = 'BRIC0000'
    elif name[0]=='Tibia' and name[1]=='Index':
        name1 = 'TIIN'

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
    elif name2=='SEBE':
        name2 = 'Belt '
    elif name2=='SPIN':
        name2 = 'T'
    elif name2=='NECK':
        name2 = 'Neck '
    elif name2=='FEMR':
        name2 = 'Femur '
    elif name2=='CLAV':
        name2 = 'Clavicle '
    elif name2=='CVEH':
        name2 = 'Veh'
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
        name2 = 'Veh ' 
        
    if name3=='00':
        name3 = ''
    elif name3=='LE':
        name3 = 'L '
    elif name3=='RI':
        name3 = 'R '
    elif name3.isnumeric():
        name3 = name3
    elif name3=='LO':
        name3 = 'Lo '
    elif name3=='UP':
        name3 = 'Up '
    elif name3=='CG':
        name3 = 'CG '
    elif name3=='TP':
        name3 = 'Top '
    
    if name4=='00':
        name4 = ''
    elif name4=='UP':
        name4 = 'Up '
    elif name4=='LO':
        name4 = 'Lo '
    elif name4=='IN':
        name4 = 'In '
    elif name4=='OU':
        name4 = 'Out '
    elif name4=='MI':
        name4 = 'Mid '
        
    if name5=='B3':
        name5 = 'Shoulder '
    elif name5=='B6':
        name5 = 'Lap '
    else:
        name5 = ''
    
    if name6=='AC':
        name6 = 'Ac'
    elif name6=='FO':
        name6 = 'F'
    elif name6=='MO':
        name6 = 'M'
    elif name6=='AN':
        name6 = 'An'
    elif name6=='AV':
        name6 = 'AV'
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