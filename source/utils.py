############################################################
#   Dev: Josias Moukpe
#   Class: Machine Learning
#   Date: 2/22/2022
#   file: utils.py
#   Description: utility functions for the genetic algorithm
#############################################################

from math import log2 

def lg(x):
    '''log2 of x with support of 0'''
    if x == 0:
        return 0
    else:
        return log2(x)

