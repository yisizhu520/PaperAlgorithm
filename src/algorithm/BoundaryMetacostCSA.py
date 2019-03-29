import algorithm.BoundaryCalculation as BoundaryCalculation
import algorithm.MetaCost as MetaCost
import algorithm.CSA as CSA

# TODO extract M to parameters
M = 5


def generate_population(training_set, classes, size, parameters):
    init_antibodies_set = []
    # TODO 顺序不合理，每个算法里头还是调用原来的反向选择算法
    for i in range(M):
        # Step5：重复执行2 - 4的过程M次，得到克隆选择过程的初始解空间
        meta_antis = MetaCost.generate_population(training_set, classes, size, parameters)
        init_antibodies_set.append(meta_antis)
    csa_antis = CSA.get_best_population_with_csa(init_antibodies_set, training_set, classes, size, parameters)
    bound_antis = BoundaryCalculation.generate_population_with_antibodies(csa_antis, training_set, parameters)
    return bound_antis
