from algorithm.DataProcessor import *
import algorithm.BoundaryCalculation as BoundaryCalculation
import algorithm.NegativeSelection as NegativeSelection

parameters = get_default_parameters()
parameters['pop_min_size'] = 200
parameters['pop_size_increment'] = 100
parameters['pop_max_size'] = 300

output_vs_csv_and_chart(NegativeSelection.generate_population,
                        BoundaryCalculation.generate_population,
              'kddcup_1k.csv',
              'boundary_vs_negative_selection_kdd',
                        parameters)


