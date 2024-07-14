#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 13:56:07 2021

@author: sunke
"""


file1 = open("MI_TempDataLog 04_28_2021.csv", "a") # The file which is append
file2 = open("MI_TempDataLog 04_29_2021.csv", "r")

count = 0
for line in file2:
    if count != 0: # Remove the first row of the second file
        file1.write(line)
    count+=1

file1.close()
file2.close()

