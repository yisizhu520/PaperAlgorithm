from data_processor import *
import algorithm.BoundaryCalculation as BoundaryCalculation
import algorithm.NegativeSelection as NegativeSelection

parameters = get_default_parameters()

output_vs_csv_and_chart(NegativeSelection.generate_population,
                        BoundaryCalculation.generate_population,
              'nsl-kddcup_1k.csv',
              'boundary_vs_negative_selection',
                        parameters)


