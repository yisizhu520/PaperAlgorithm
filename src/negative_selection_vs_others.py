from algorithm.DataProcessor import *
import sys

parameters = {}
parameters["step_size"] = 0.01
parameters['pop_size'] = 300
parameters['iteration_count'] = 3

compare_with_other_algorithm('kddcup_1k.csv',
                             '1_vs_5_kdd', parameters)
