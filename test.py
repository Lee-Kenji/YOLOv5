import argparse
import csv
import os
import platform
import sys
from pathlib import Path
from multiprocessing import Process, Manager
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import torch

def update_matrix(xcoor_list, matrix_of_people):

    # Sort xcoor_list
    xcoor_list = np.sort(xcoor_list)
    # Deal with empty screen case
    if len(xcoor_list) == 0: 
        #print("Empty list supplied")
        update_iteration = 0
        for i in matrix_of_people[-2,:]:
            if (i != 0) or (matrix_of_people[0,update_iteration] != 0):
                matrix_of_people[-2,update_iteration] = matrix_of_people[-2,update_iteration] + 1
            update_iteration = update_iteration + 1
        index_count = 0
        for i in matrix_of_people[-2,:]:
            if i >= 3:
                matrix_of_people[:, index_count] = 0
                #print("Column",index_count,"cleared")
            index_count = index_count + 1
        return matrix_of_people
    
    # Move matrix down by 1 for new coordinates to update
    for i in range(len(matrix_of_people[:,0]) - 3):
        matrix_of_people[9-i,:] = matrix_of_people[8-i,:]
    matrix_of_people[0,:] = 0

    # Assume that the xcoor_list is longer than previously recorded in matrix, find the outlier (new person) coordinate
    outlier_matrix = np.zeros((len(xcoor_list), np.count_nonzero(matrix_of_people[1,:])), dtype=float)
    #print("OUTLIER MATRIX BEFORE")
    #print(outlier_matrix)
    if len(xcoor_list) > np.count_nonzero(matrix_of_people[1,:]):
        countI = 0
        for i in xcoor_list:
            countJ = 0
            for j in np.nonzero(matrix_of_people[1,:])[0]:
                outlier_matrix[countI, countJ] = abs(i - j)
                countJ = countJ + 1
            countI = countI + 1
        #print("OUTLIER MATRIX AFTER")
        #print(outlier_matrix)
  

    
        # Elimination of outliers
        for i in range(np.count_nonzero(matrix_of_people[1,:])):
            min_index = np.argmin(outlier_matrix)
            min_row, min_col = np.unravel_index(min_index, outlier_matrix.shape)
            outlier_matrix[min_row,:] = 3
        #print("OUTLIER MATRIX EDITED")
        #print(outlier_matrix)
        removing_numbers = []
        for i in range(len(xcoor_list) - np.count_nonzero(matrix_of_people[1,:])):
            if np.count_nonzero(outlier_matrix):
                min_index = np.argmin(outlier_matrix)
                min_row, min_col = np.unravel_index(min_index, outlier_matrix.shape)
                outlier_matrix[min_row,:] = 3
                removing_numbers.append(xcoor_list[min_row])
                for j in range(len(matrix_of_people[0,:])):
                    if (matrix_of_people[1,j] == 0) and (matrix_of_people[0,j] == 0):
                        matrix_of_people[0,j] = xcoor_list[min_row]
                        #print("UPDATED NO:", xcoor_list[min_row])
                        break
        
        # remove the extra numbers from xcoor_list before proceeding for the matching
        xcoor_list = [x for x in xcoor_list if x not in removing_numbers]
        #print("LENGTH OF XCOORLIST:", len(xcoor_list))
            



    # Assign closest coordinate to its corresponding index
    taken_slot = np.zeros((len(matrix_of_people[0,:])), dtype=float)
    for i in range(len(xcoor_list)):
        differences = np.abs(xcoor_list[i] - matrix_of_people[1, :])
        for j in range(len(matrix_of_people[0,:])):
            if ((matrix_of_people[1,j] == 0) and (len(xcoor_list) <= np.count_nonzero(matrix_of_people[1,:])) or (taken_slot[j] == 1)):
                differences[j] = 3

        min_difference_index = np.argmin(differences)
        taken_slot[min_difference_index] = 1
        matrix_of_people[0, min_difference_index] = xcoor_list[i]
        #print("INPUTTING: ", xcoor_list[i])
        #print(matrix_of_people)
    # Check for missing frames / coordinates, update error count and paste value to remove motion change
    for i in range(len(matrix_of_people[0,:])):
        if (matrix_of_people[0,i] == 0) and (matrix_of_people[1,i] != 0):
            matrix_of_people[-2,i] = matrix_of_people[-2,i] + 1
            matrix_of_people[0,i] = matrix_of_people[1,i]

    # Check for over certain amount of repetition before deleting the entire column
    index_count = 0
    for i in matrix_of_people[-2,:]:
        if i >= 3:
            matrix_of_people[:, index_count] = 0
            #print("Column",index_count,"cleared")
        index_count = index_count + 1


    return matrix_of_people


xcoor_list = [0.5]
#xcoor_list = []

matrix_of_people = matrix_of_people = np.zeros((12, 11), dtype=float)

#matrix_of_people[0,2] = 0.9
#matrix_of_people[0,4] = 0.6
#matrix_of_people[0,3] = 0.2
print("BEFORE")
print(matrix_of_people)
matrix_of_people = update_matrix(xcoor_list, matrix_of_people)

#matrix_of_people[-3,3] = 69
#matrix_of_people[-2,3] = 3
#matrix_of_people[-3,2] = 420
#print(matrix_of_people)
#matrix_of_people = clear_col(matrix_of_people)
print("AFTER")
print(matrix_of_people)