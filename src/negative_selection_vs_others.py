from algorithm.DataProcessor import *
import sys

parameters = {}
parameters["step_size"] = 0.01
parameters['pop_size'] = 400
parameters['iteration_count'] = 3

compare_with_other_algorithm('nsl-kddcup_1k.csv',
                             '1_vs_4', parameters)
