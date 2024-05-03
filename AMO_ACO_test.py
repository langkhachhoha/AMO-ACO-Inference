# import torch
# from torch.distributions import Categorical
# import random
# import itertools
# import numpy as np
# import copy
# import torch
# from torch import nn
# from torch.nn import functional as F
# from copy import deepcopy
# import math 
# import numpy as np
# from Model.Config import *
# from Model.AMO import * 
# from Ant_k_starts import *
# from Normalize_data import *
# from Ant import *
# from Injection import *
# from Cross_Exchange import *
# # from Local_search import * 
from Model.Config import *
from Draw import *
import argparse
import os 





def get_args():
    parser = argparse.ArgumentParser(description="CVRPTW_GECCO2024")
    parser.add_argument("--size", type=int, default=100, help="size of data")
    parser.add_argument("--data_path", type=str, default='None', help="the folder data to run")
    parser.add_argument("--epochs", default=10, type=int, help="Total number of epochs")

    args = parser.parse_args()
    return args

if __name__ == '__main__' :
    args = get_args()
    device = None
    Data_to_run = []
    if args.size == 100:
        cfg = Data_100()
        Data_to_run = []
        for filename in os.listdir('txt/'):
            if len(filename) <= 9 and filename[:3] not in ['toy', '.DS']:
                Data_to_run.append(filename)

    if args.size == 200:
        cfg = Data_200()
        Data_to_run = []
        for filename in os.listdir('txt/'):
            if filename[:4] in ['C1_2', 'C2_2', 'R1_2', 'R2_2'] or filename[:5] in ['RC1_2', 'RC2_2']:
                Data_to_run.append(filename)

    if args.size == 400:
        cfg = Data_400()
        for filename in os.listdir('txt/'):
            if filename[:4] in ['C1_4', 'C2_4', 'R1_4', 'R2_4'] or filename[:5] in ['RC1_4', 'RC2_4']:
                Data_to_run.append(filename)


    if args.data_path != 'None':
        Data_to_run = []
        Data_to_run.append(args.data_path)

    print(Data_to_run)


    # from Local_search import * 



    for da in Data_to_run:
        with open('txt/{}'.format(da), 'r') as source:
            content = source.read()

        with open('Data.txt', 'w') as destination:
            destination.write(content)
        import torch
        from torch.distributions import Categorical
        import random
        import itertools
        import numpy as np
        import copy
        import torch
        from torch import nn
        from torch.nn import functional as F
        from copy import deepcopy
        import math 
        import numpy as np
        # from Model.Config import *
        from Model.AMO import * 
        from Ant_k_starts import *
        from Normalize_data import *
        from Ant import *
        from Injection import *
        from Cross_Exchange import *
        from Local_search import * 
        from Draw import *


        EPS = 1e-10
        model = Net3().to(device)
        model.load_state_dict(torch.load('AMO_ACO_{}.pt'.format(cfg.graph_size), map_location=torch.device('cpu')))

        pyg_data_normalize = normalize_data(cfg)
        heuristic_measure, log, topk = model(pyg_data_normalize)
        heuristic_measure = heuristic_measure.reshape((cfg.graph_size+1, cfg.graph_size+1))



        max_cap, xcoord, ycoord, demand, e_time, l_time, s_time, data = load_data()

        data = torch.tensor([[float(x) for x in y] for y in data])
        tsp_coordinates = data[:, 1:3] 
        demands = torch.tensor(demand, dtype = torch.float32)
        time_window = data[:, 4:]
        durations = time_window[:, -1] 
        distances = gen_distance_matrix(tsp_coordinates, device = device)

        aco = ACO(distances, demands, time_window, 10, topk, max_cap, heuristic=heuristic_measure, n_ants=cfg.n_ants)

        max_cap, xcoord, ycoord, demand, e_time, l_time, s_time, data = load_data()

        CAP=max_cap
        colony=Ant(data,CAP,0.7, heuristic_measure)
        colony.customer_cord()
        colony.euclidean_distance()
        colony.width_window()
        _ = colony.path_pheromon()


        def convert_dict(path): # path: tensor depot 0(m,)
            path += 1
            path = path.to(torch.long)
            zero_indices = torch.where(path == 1)[0]
            zero_indices = zero_indices.tolist()
            path = path.tolist()
            while zero_indices[-1] - zero_indices[-2] == 1:
                zero_indices.pop()
            dict = {}
            for i in range(len(zero_indices) - 1):
                dict[i] = path[zero_indices[i]: zero_indices[i+1] + 1 ]
            return dict

        def update_btnt(path, cost, pheromone, effort, k_candidate, prob_to_update, reward_coff, best_cost):
            '''
            path: Loi giai tiem nang
            cost: ham cost cua loi giai
            pheromone: aco.pheromone
            effort: so iter truoc khi ket thuc
            '''
            if cost < best_cost:
                reward = int(k_candidate * prob_to_update * reward_coff) * effort
            else:
                reward = int(k_candidate * prob_to_update) * effort
            for route in path.values(): # route
                for l in range(len(route) - 1):
                    pheromone[int(route[l] - 1)][int(route[l+1] - 1)] += reward/cost
            reward_coff += 2
            return reward_coff, pheromone


        import time
        max_iteration = args.epochs 
        n_customer = len(data)
        k_candidate = 100
        prob_to_update = 0.05
        best = 1e10
        best_cost = 1e10
        aco.decay = 1
        aco.beta = 0.5
        elitism_set = 0
        reward_coff = 2
        effort = 0
        final_path = 0
        prob_to_destroy = 0.05
        cnt = 0
        counter = 0
        best_data = []
        best_cost_data = []
        destroy_data = []
        t1 = time.time()


        def save_solution(ants_route, travel_distance, BTNT, max_route, max_travel):
            if len(ants_route.keys()) < max_route:
                BTNT[-1] = ants_route
                max_travel = travel_distance
                max_route = len(ants_route.keys())
            elif len(ants_route.keys()) == max_route and travel_distance < max_travel:
                BTNT[-1] = ants_route
                max_travel = travel_distance
                max_route = len(ants_route.keys())
            return BTNT, max_travel, max_route


        BTNT = [0]
        max_travel = 1e10

        max_route = int(n_customer)
        for k in range(max_iteration):

            paths, costs = aco.run()
            '''
            TO DO:
            - Chon 100 paths co gia tri tot nhat # DONE 
            - Xay ham chuyen paths ve dinh dang dictionary nhu BTNT_IBSO_ACO # DONE
            - Mode colony # DONE
            - Thuc hien Injection va Cross Exchange # DONE 
            - Thuc hien Local Search ngau nhien # DONE
            - Xac suat lua chon < 0.3 # DONE
            - continue
            '''
            local_path = []
            candidate_values, indexs = torch.topk(costs, k = k_candidate, largest=False)
            candidate_path = paths.T[indexs] # (k * prob_size)
            for i, (value, path) in enumerate(zip(candidate_values, candidate_path)):
                ants_route = convert_dict(path)
                if torch.rand(1) < prob_to_update:
                    travel_distance, ants_route = injection(ants_route, colony, 0.5)
                    travel_distance, ants_route = cross_exchange(ants_route, colony)
                    travel_distance, ants_route = ls_1route(ants_route, colony)
                    travel_distance, ants_route = ls3(ants_route)
                    BTNT, max_travel, max_route = save_solution(ants_route, travel_distance, BTNT, max_route, max_travel)
                    candidate_values[i] = travel_distance
                local_path.append(ants_route)

            aco.pheromone *= aco.decay
            value_to_update, index_to_update = torch.topk(candidate_values, k = int(k_candidate * prob_to_update), largest=False)
            for i, j in enumerate(index_to_update):
                path = local_path[j] # dict
                for route in path.values(): # route
                    for l in range(len(route) - 1):
                        if i == 0:
                            aco.pheromone[int(route[l] - 1)][int(route[l+1] - 1)] += 1/value_to_update[i]
                        aco.pheromone[int(route[l] - 1)][int(route[l+1] - 1)] += 1/value_to_update[i]
            if torch.min(candidate_values) < best:
                best = torch.min(candidate_values)
                best_path = local_path[index_to_update[0]]
                if k > 0 and elitism_set != 0:
                    path, cost = elitism_set
                    reward_coff, pheromone = update_btnt(path, cost, aco.pheromone, effort, k_candidate, prob_to_update, reward_coff, best_cost)
                    aco.pheromone = pheromone 
                elitism_set = (best_path, best)  # tuple (path, cost)
                effort = 1
                tries = 0

            # Alimentation:
            if tries == 3 and elitism_set != 0:
                tries = 0
                path, cost = elitism_set
                reward_coff, pheromone = update_btnt(path, cost, aco.pheromone, effort, k_candidate, prob_to_update, reward_coff, best_cost)
                elitism_set = 0
                aco.pheromone = pheromone 


            if elitism_set == 0:
                counter += 1


            if elitism_set != 0:
                counter = 0
                ants_route, cost = elitism_set
                for _ in range(1):
                    travel_distance, ants_route = injection(ants_route, colony, 0.5)
                for _ in range(1):
                    travel_distance, ants_route = cross_exchange(ants_route, colony)
                travel_distance, ants_route = local_search(ants_route, colony, n_customer, cfg.q)
                travel_distance, ants_route = ls3(ants_route)
                for _ in range(50):
                    travel_distance, ants_route = ls_1route(ants_route, colony)
                BTNT, max_travel, max_route = save_solution(ants_route, travel_distance, BTNT, max_route, max_travel)
                if travel_distance < cost:
                   elitism_set = (ants_route, travel_distance)
                   effort += 1
                if travel_distance < best_cost:
                    best_cost = travel_distance 
                    final_path = ants_route
                else:
                   effort += 1
                   tries += 1

            if elitism_set == 0 and counter > prob_to_destroy * max_iteration:
            # if 1:
                print(destroy)
                travel_distance, ants_route = destroy(final_path, colony, 0.5)
                travel_distance, ants_route = destroy_ls_1route(final_path, colony)
                r = np.random.random()
                if r < 0.5:
                    travel_distance, ants_route = ls1(final_path)
                else:
                    travel_distance, ants_route = ls2(final_path)
                print("Destroy: ", travel_distance)
                elitism_set = (ants_route, travel_distance)
                counter = 0
                destroy_data.append(k)








            best_data.append(best)
            best_cost_data.append(best_cost)
            print('epoch {}: Best: {}, Alimentation: {}'.format(k, best, best_cost))
        t2 = time.time()
        time_run = t2-t1


        for _ in range(10):
            ants_route, cost = BTNT[-1], max_travel
            for _ in range(1):
                travel_distance, ants_route = injection(ants_route, colony, 0.5)
            for _ in range(1):
                travel_distance, ants_route = cross_exchange(ants_route, colony)
            travel_distance, ants_route = local_search(ants_route, colony, n_customer, cfg.q)
            travel_distance, ants_route = ls3(ants_route)
            for _ in range(50):
                travel_distance, ants_route = ls_1route(ants_route, colony)
            BTNT, max_travel, max_route = save_solution(ants_route, travel_distance, BTNT, max_route, max_travel)
            print(max_travel, max_route)



        if 1000*max_route + max_travel > 1000*len(final_path.keys()) + best_cost:
            final_route = final_path
            final_cost = best_cost
            final_vehicle = len(final_path.keys())
        else:
            final_route = BTNT[-1]
            final_cost = max_travel
            final_vehicle = max_route


        with open('Data.txt', 'r') as f:
            name = f.readline()
        name.rstrip()

        def change_2_save(index, route):
            ans = "Route #{}:".format(index)
            for node in route[1:-1]:
                ans += " " + str(node-1)
            return ans 

        with open('Solution.txt', 'a') as f:
            f.write(name + '\n')
            for id, i in enumerate(final_route.values()):
                f.write(change_2_save(id+1, i) + '\n')
            f.write("Time: {}".format(round(time_run,2)) + '\n')
            f.write("Vehicle: {}".format(final_vehicle)+ '\n')
            f.write("Distance: {}".format(final_cost)+ '\n')
            f.write("-----------------------------------------------"+ '\n')


