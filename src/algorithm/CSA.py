from algorithm.NegativeSelection import *


def get_best_population_with_csa(init_antibodies_set, training_set, classes, size, parameters, chooses_ratio,
                                 clone_multiple_count):
    iteration_time = size
    save_count = 5
    accuracy = 0.000001
    last_results = []
    results = fill_fitness_for_antibodies_set(init_antibodies_set, training_set)
    for i in range(iteration_time):
        results = csa(results, training_set, classes, parameters, chooses_ratio, clone_multiple_count)
        results.sort(key=lambda result: result['fitness'], reverse=True)
        last_results.append(results[0])
        if len(last_results) == save_count:
            flag = True
            for j in range(save_count - 1):
                if abs(last_results[save_count - 1]['fitness'] - last_results[j]['fitness']) > accuracy:
                    flag = False
                    break
            if flag:
                # print(last_results[save_count - 1]['antibodies'])
                # print(results[0]['fitness'])
                return last_results[save_count - 1]['antibodies']
            else:
                last_results.remove(last_results[0])


def fill_fitness_for_antibodies_set(antibodies_set, training_set):
    result_objs = []
    for i in range(len(antibodies_set)):
        antibodies = antibodies_set[i]
        fitness = cal_fitness(antibodies, training_set)
        obj = {'antibodies': antibodies, 'fitness': fitness}
        result_objs.append(obj)
    return result_objs


def csa(population_results, training_set, classes, parameters, chooses_ratio, clone_multiple_count):
    choose_top_size = int(len(population_results) * chooses_ratio)
    clone_size = clone_multiple_count
    result_objs = population_results
    result_objs.sort(key=lambda result: result['fitness'], reverse=True)
    top_result = result_objs[:choose_top_size]
    last_result = result_objs[choose_top_size:len(result_objs)]
    # Step7：以选择率S选择前S个解，并以C倍克隆前S个解，以1-亲和度为变异率进行变异；
    clone_pop_results = clone_population(top_result, clone_size)
    for pop_result in clone_pop_results:
        antis = pop_result['antibodies']
        random_reproduced_antibody_percent = 1 - pop_result['fitness']
        antis = reproduce_antibodies(antis, random_reproduced_antibody_percent, training_set, classes, parameters)
        pop_result['fitness'] = cal_fitness(antis, training_set)

    for pop_result in last_result:
        clone_pop_results.append(pop_result)
    clone_pop_results.sort(key=lambda result: result['fitness'], reverse=True)
    last_result = clone_pop_results[:len(last_result)]

    combine_result = top_result + last_result
    # print(combine_result)
    return combine_result


def cal_fitness(antibodies, data_set):
    # Step6：计算解空间每一组解的Cover和1 - Over的值，其中Cover和Over分别取所有类对应值的均值；以此值作为亲和度，对解空间的解进行降序排序
    data_hit_dict = {}
    anti_hit_dict = {}
    for i in range(len(data_set)):
        data_hit_dict[i] = []
    for j in range(len(antibodies)):
        anti_hit_dict[j] = []

    for i in range(len(data_set)):
        d = data_set[i]
        for j in range(len(antibodies)):
            a = antibodies[j]
            dis = distance(d[1], a[1])
            if dis <= a[2]:
                anti_hit_dict[j].append(d)
                data_hit_dict[i].append(a)

    Vover = 0
    Vself = 0
    Over = 0
    Cover = cal_cover(antibodies, data_set)
    for i in range(len(antibodies)):
        repeat_data_count = 0
        for j in range(i, len(antibodies)):
            intersection = [val for val in anti_hit_dict[i] if val in anti_hit_dict[j]]
            repeat_data_count += len(intersection)
        # 公式3-13
        Pi = 0
        if len(anti_hit_dict[i]) != 0:
            Pi = repeat_data_count / float(len(anti_hit_dict[i]))
        Vi = cal_Vi(len(antibodies[i][1]), antibodies[i][2])
        Vover += Pi * Vi
        Vself += Vi * (1 - Pi)
    Vself = Vself / Cover
    if Vself != 0:
        Over = Vover / Vself
    return (Cover + 1 - Over) / 2


def cal_Vi(n, r):
    return (math.pow(math.pi, n / 2) * math.pow(r, n)) / math.gamma(n / 2 + 1)


def cal_over():
    return


def cal_cover(antibodies, data_set):
    hit_count = 0
    for d in data_set:
        for a in antibodies:
            dis = distance(d[1], a[1])
            if dis <= a[2]:
                hit_count = hit_count + 1
                break
    return float(hit_count) / float(len(data_set))


def clone_population(original_pops, clone_size):
    results = []
    for pop in original_pops:
        for i in range(clone_size):
            results.append(copy.deepcopy(pop))
    return results


def reproduce_antibodies(antibodies, percent, training_set, classes, parameters):
    classes_antibody_dict = get_class_antibody_dict(antibodies)
    # remove percent of antibodies randomly
    for cls in classes_antibody_dict:
        antis = classes_antibody_dict[cls]
        remove_size = int(len(antis) * percent)
        for i in range(remove_size):
            antis.remove(choice(antis))
    # reproduce percent of antibodies
    reproduced_pops = generate_population(training_set, classes, int(len(antibodies) * percent), parameters)
    new_antis = []
    for cls in classes_antibody_dict:
        for a in classes_antibody_dict[cls]:
            new_antis.append(a)
    for pop in reproduced_pops:
        new_antis.append(pop)
    return new_antis
