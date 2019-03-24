from algorithm.DataProcessor import *
import algorithm.MetaCost as Metacost
import algorithm.NegativeSelection as NegativeSelection

parameters = get_default_parameters()

output_vs_csv_and_chart(NegativeSelection.generate_population,
                        Metacost.generate_population,
                        'nsl-kddcup_1k.csv',
                        'metacost_vs_negative_selection',
                        parameters)
