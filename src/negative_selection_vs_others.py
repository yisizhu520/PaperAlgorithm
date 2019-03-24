from algorithm.DataProcessor import *

parameters = {}
parameters["step_size"] = 0.1
parameters['pop_size'] = 200
parameters['iteration_count'] = 3

compare_with_other_algorithm('nsl-kddcup_1k.csv',
                             '1_vs_4', parameters)
