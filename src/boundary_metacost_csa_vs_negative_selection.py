from data_processor import *
import algorithm.BoundaryMetacostCSA as BoundaryMetacostCSA
import algorithm.NegativeSelection as NegativeSelection

parameters = get_default_parameters()

output_vs_csv_and_chart(NegativeSelection.generate_population,
                        BoundaryMetacostCSA.generate_population,
              'nsl-kddcup_1k.csv',
              'boundary_metacost_csa_vs_negative_selection',
                        parameters)


