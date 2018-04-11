# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 13:47:02 2018

for image processing

@author: tangk
"""
import imageio 

def write_tif_stack(filepath,files,image_range):
    for j in range(len(filepath)):
#        print(filepath[j])
#        print(files[j])
        reader = imageio.get_reader(filepath[j])
        writer = imageio.get_writer(files[j] + '.tif')
        for i in image_range:
            writer.append_data(reader.get_data(i))
        writer.close()