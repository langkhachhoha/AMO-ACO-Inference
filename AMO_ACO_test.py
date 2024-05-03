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


import matplotlib.pyplot as plt
import numpy as np
import math
import random
import copy
from scipy.spatial.distance import cdist
from ultis import *
from Ant import Ant 
# from Local_search import *

def route_1_ver2(routes):
    route=copy.deepcopy(routes)
    for i in range(len(route)):
            route[i]-=1
    return route

def find_route_target(customer, ants_route, colony, p, index):
    lim = p * max(colony.distance_matrix.values())
    route_target = []
    for key, value in ants_route.items():
        if key != index:
            check = 1
            for customer_target in value[1:-1]:
                if colony.distance_matrix[customer, customer_target] > lim:
                    check = 0
                    break
            if check:
                route_target.append(key)
    return route_target



def injection(ants_route, colony, p):
    min_route = 1000
    index = 0
    for key, value in ants_route.items():
        if len(value) < min_route and len(value) > 2:
            index = key
            min_route = len(value)
    colony_distance = 0
    for value in ants_route.values():
        colony_distance += caculate_distance(value, colony)
    ants_route_copy = {key: copy.deepcopy(value) for key, value in ants_route.items()}
    select = []
    for customer in ants_route[index][1:-1]:
        route_target = find_route_target(customer, ants_route_copy, colony, p, index)
        for route in route_target:
            done = 0
            new_route = check_feasible(customer, ants_route_copy[route], colony)
            if new_route != []:
                ants_route_copy[route] = new_route
                select.append(customer)
                done = 1
                break


            else:
                for customer_target in ants_route_copy[route][1:-1]:
                    new_route = ants_route_copy[route].copy()
                    new_route.remove(customer_target)
                    r = check_feasible(customer, new_route, colony)
                    if r == []:
                        continue

                    check = 0

                    for key, value in ants_route_copy.items():
                        if key != index and key != route:
                            test = check_feasible(customer_target, value, colony)
                            if test != []:
                                ants_route_copy[key] = test 
                                check = 1
                                break 
                    if check:
                        ants_route_copy[route] = r 
                        # select.append(customer)
                        done = 1
                        break 
            if done:
                select.append(customer)
                break 



    for i in select:
        ants_route_copy[index].remove(i)
    
    travel_distance = 0
    for route in ants_route_copy.values():
        for i in range(len(route)-1):
            travel_distance += colony.distance_matrix[route[i], route[i+1]]
    if travel_distance < colony_distance:
        return travel_distance, change(ants_route_copy)
    
    return colony_distance, change(ants_route)

def ls_1route(ants_route, colony):
    index = np.random.randint(0, len(ants_route.values()))
    route = ants_route[index]
    colony_distance = caculate_distance(route, colony)
    distance = 0
    for value in ants_route.values():
        distance += caculate_distance(value, colony)
    for i in range(1, len(route) - 1):
        test_route = copy.deepcopy(route)
        test_route.remove(route[i])

        for j in range(1, len(test_route) - 1):
            test_route_2 = copy.deepcopy(test_route)
            test_route_2.insert(j, route[i])
            if check(route_1_ver2(test_route_2)):
                if caculate_distance(test_route_2, colony) < colony_distance:
                    ants_route_copy = {key: copy.deepcopy(value) for key, value in ants_route.items()}
                    ants_route_copy[index] = test_route_2
                    distance = 0
                    for value in ants_route_copy.values():
                        distance += caculate_distance(value, colony)
                    return distance, change(ants_route_copy)
    return distance, change(ants_route)

    
def destroy_ls_1route(ants_route, colony):
    ants_route_copy = {key: copy.deepcopy(value) for key, value in ants_route.items()}
    indexs = np.random.choice(np.arange(len(ants_route.values())), size=3, replace=True)
    for index in indexs:
        route = ants_route[index]
        for i in range(1, len(route) - 1):
            test_route = copy.deepcopy(route)
            test_route.remove(route[i])
            for j in range(1, len(test_route) - 1):
                test_route_2 = copy.deepcopy(test_route)
                test_route_2.insert(j, route[i])
                if check(route_1_ver2(test_route_2)):
                        ants_route_copy[index] = test_route_2
    colony_distance = 0
    for value in ants_route_copy.values():
        colony_distance += caculate_distance(value, colony)
    return colony_distance, change(ants_route_copy)





def destroy(ants_route, colony, p):
    index = np.random.randint(0, len(ants_route.values()))
    colony_distance = 0
    for value in ants_route.values():
        colony_distance += caculate_distance(value, colony)
    ants_route_copy = {key: copy.deepcopy(value) for key, value in ants_route.items()}
    select = []
    for customer in ants_route[index][1:-1]:
        route_target = find_route_target(customer, ants_route_copy, colony, p, index)
        for route in route_target:
            done = 0
            new_route = check_feasible(customer, ants_route_copy[route], colony)
            if new_route != []:
                ants_route_copy[route] = new_route
                select.append(customer)
                done = 1
                break


            else:
                for customer_target in ants_route_copy[route][1:-1]:
                    new_route = ants_route_copy[route].copy()
                    new_route.remove(customer_target)
                    r = check_feasible(customer, new_route, colony)
                    if r == []:
                        continue

                    check = 0

                    for key, value in ants_route_copy.items():
                        if key != index and key != route:
                            test = check_feasible(customer_target, value, colony)
                            if test != []:
                                ants_route_copy[key] = test 
                                check = 1
                                break 
                    if check:
                        ants_route_copy[route] = r 
                        done = 1
                        break 
            if done:
                select.append(customer)
                break 



    for i in select:
        ants_route_copy[index].remove(i)
    
    travel_distance = 0
    for route in ants_route_copy.values():
        for i in range(len(route)-1):
            travel_distance += colony.distance_matrix[route[i], route[i+1]]
    return travel_distance, change(ants_route_copy)


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


    import matplotlib.pyplot as plt
    import numpy as np
    import math
    import random
    import copy
    from scipy.spatial.distance import cdist
    from ultis import *
    from Ant import Ant 
    from Load_data import *
    import time 




    def check(route):
        # max_cap, xcoord, ycoord, demand, e_time, l_time, s_time, data = data_zip
        route2 = copy.deepcopy(route)

        # check cap
        cap = 0 
        for x in route2:
            cap += demand[x]
        if cap>max_cap: 
            return False

        #check time
        cur_time=0
        for i in range(len(route2)-1):
            cur_time=cur_time+s_time[route2[i]]+distance(route2[i],route2[i+1])
            if cur_time<e_time[route[i+1]]:
                cur_time=e_time[route2[i+1]]
            if cur_time>l_time[route2[i+1]]:
                return False
        return True

    def distance(i,j): #tính khoảng cách 2 điểm
        # max_cap, xcoord, ycoord, demand, e_time, l_time, s_time, data = load_data()
        # max_cap, xcoord, ycoord, demand, e_time, l_time, s_time, data = data_zip
        return ((xcoord[i]-xcoord[j])**2+(ycoord[i]-ycoord[j])**2)**(1/2)

    def cost2(route):  # tính tổng đường đi của 1 cá thể
        if route[0]!=-1:
            sum=0
            for i in route:
                for j in range(0,len(i)-1):
                    sum+=distance(int(i[j]),int(i[j+1]))
            return sum
        else:
            return float('inf')

    # (-1)
    def route_1(routes):
        route=copy.deepcopy(routes)
        for i in range(len(route)):
            for j in range(len(route[i])):
                route[i][j]-=1
        return route

    def route_1_ver2(routes):
        route=copy.deepcopy(routes)
        for i in range(len(route)):
                route[i]-=1
        return route



    # (+1)
    def route__1(routes):
        route=copy.deepcopy(routes)
        for i in range(len(route)):
            for j in range(len(route[i])):
                route[i][j]+=1
        return route




    def search(route):
        route=route_1(route)
        a = len(route)
        lst = []
        route_id = np.random.choice(np.arange(a), size = a, replace=False)
        for i in route_id:
            lst.append(route[i])
        route = lst 
        for i in range(len(route)-1):
            for j in range(i+1,len(route)):
                for k in range(1,len(route[i])-1):
                    for t in range(1,len(route[j])-1):
                            new_route=copy.deepcopy(route)
                            z=new_route[i][k]
                            new_route[i][k]=new_route[j][t]
                            new_route[j][t]=z
                            if check(new_route[i]) and check(new_route[j]) and cost2(new_route)< cost2(route):
                                z=route[i][k]
                                route[i][k]=route[j][t]
                                route[j][t]=z
        while [0,0] in route:
            route.remove([0,0])
        return route__1(route)

    def search2(routes,colony,q):
        # routes=route_1(routes)
        routes=route_1(routes)
        a = len(routes)
        lst = []
        route_id = np.random.choice(np.arange(a), size = a, replace=False)
        for i in route_id:
            lst.append(routes[i])


        routes = lst 
        m = len(routes)
        matrix = central_3(routes, colony)
        p = int(q * m) 
        if p == 0:
            matr = matrix[:, :1]
        else:
            matr = matrix[:, :p]
        while routes[-1]==[0,0]:
            routes.pop()
        r = np.random.random()
        if r <0.1:
            matr = np.array([i[::-1] for i in matr])
        for i in range(len(routes)):
            cnt = 0
            if routes[i] == [0,0]:
                continue 
            for j in matr[i]:
                cnt += 1
                if i!=j and routes[j] != [0,0]:
                    k=1
                    while (k<len(routes[i])-1):
                        for t in range(1,len(routes[j])-1):
                          if k<len(routes[i])-1:
                            new_route=copy.deepcopy(routes)
                            z=new_route[i][k]
                            new_route[i].pop(k)
                            new_route[j].insert(t,z)
                            if cost2(new_route)< cost2(routes) and check(new_route[j]):
                                routes[i].pop(k)
                                routes[j].insert(t,z)


                        k+=1
        while [0,0] in routes:
            routes.remove([0,0])
        return route__1(routes)







    def search4(route, colony, n_customer):
        lst = []
        cus = []
        for key, value in enumerate(route):
            distance = 0
            for i in range(len(value)-1):
                distance += colony.distance_matrix[value[i], value[i+1]]
            lst.append(distance)
        route = route_1(route)

        r = np.random.randint(1,6)
        if r < 4:
            # print(1)
            if np.random.random() < 0.5:
                select_1, select_2, select_3 = np.argsort(np.array(lst))[-3:]
                selected_route = [copy.deepcopy(route[select_1]) ,
                          copy.deepcopy(route[select_2]),
                          copy.deepcopy(route[select_3])]
                a = [select_1, select_2, select_3]
            else:
                select_1, select_2, select_3 = np.random.choice(np.arange(0, len(route)), size=3, replace=False)
                selected_route = [copy.deepcopy(route[select_1]) ,
                          copy.deepcopy(route[select_2]),
                          copy.deepcopy(route[select_3])]
                a = [select_1, select_2, select_3]




        else:
            # print(3)
            select_1, select_2 = central(route, colony)
            selected_route = [copy.deepcopy(route[select_1]) ,
                          copy.deepcopy(route[select_2])
                          ]
            a = [select_1, select_2]


        if r<4 and len(route[select_1]) + len(route[select_2]) + len(route[select_3]) > n_customer/2:
            if np.random.random() < 0.5:
                select_1, select_2 = central(route, colony)
                selected_route = [copy.deepcopy(route[select_1]) ,
                              copy.deepcopy(route[select_2])
                              ]
                a = [select_1, select_2]
            else:
                return route__1(route)

        for i in range(len(selected_route)-1): # Bắt đầu từ route_1
            for j in range(i+1,len(selected_route)): # Lặp qua các route tiếp theo
                for k1 in range(1,len(selected_route[i])-2): # Lặp qua các tp ở route 1
                  for k2 in range(k1+1,len(selected_route[i])-1): # 
                    for t1 in range(1,len(selected_route[j])-2):
                      for t2 in range(t1+1,len(selected_route[j])-1):
                            new_route=copy.deepcopy(selected_route)
                            zk=copy.deepcopy(new_route[i][k1:k2+1]) # Ok
                            zt=copy.deepcopy(new_route[j][t1:t2+1]) # Ok
                            del new_route[i][k1:k2+1] # Xoá
                            del new_route[j][t1:t2+1] # Xoá 
                            new_route[i]=new_route[i][:k1]+zt+new_route[i][k1:]
                            new_route[j]=new_route[j][:t1]+zk+new_route[j][t1:]
                            if cost2(new_route)< cost2(selected_route) and check(new_route[i]) and check(new_route[j]):
                                zk=copy.deepcopy(selected_route[i][k1:k2+1])
                                zt=copy.deepcopy(selected_route[j][t1:t2+1])
                                del selected_route[i][k1:k2+1]
                                del selected_route[j][t1:t2+1]
                                selected_route[i]=selected_route[i][:k1]+zt+selected_route[i][k1:]
                                selected_route[j]=selected_route[j][:t1]+zk+selected_route[j][t1:]
                                for key, value in enumerate(selected_route):
                                    route[a[key]] = value 
                                return route__1(route)
        while [0,0] in route:
            route.remove([0,0])


        return route__1(route)




    def local_search(t, colony, n_customer, q):
        t1=copy.deepcopy(t)
        routes=[]
        # routes = t1
        for i in range(len(t1)):
            routes.append(t1[i])
        routes=search4(search2(search(routes), colony, q), colony, n_customer)
        index=0
        result={}
        for x in (routes):
            if x!=[1,1]:
                result[index]=x
                index+=1
        return cost2((route_1(routes))), result

    # Destroy Chinh
    def cost_route(route):  # tính tổng đường đi của 1 cá thể
        sum=0
        for i in range(len(route)-1):
            sum+=distance(int(route[i]),int(route[i+1]))
        return sum

    def ls1(t,num_slices=10):
        t1=copy.deepcopy(t)
        routes=[]
        # routes = t1
        for i in range(len(t1)):
            routes.append(t1[i])

        routes=route_1(routes)
        while routes[-1]==[0,0]:
            routes.pop()
        index = random.sample(range(0, len(routes)), 3)
        # print(index)
        new_route=[[],[],[]]
        change=[[0,1],[1,2],[2,0]]
        for i in range(0,3):
            cut=int((len(routes[index[i]])-2)/num_slices)
            # print(cut)
            cut1=len(routes[index[i]])-2-num_slices*cut
            # print(cut1)
            t=1
            for j in range(cut1):
                new_route[i].append(routes[index[i]][t:t+cut+1])
                t+=cut+1
            for j in range(num_slices-cut1):
                new_route[i].append(routes[index[i]][t:t+cut])
                t+=cut
        for i in range(0,num_slices):
            best=100000000
            best_choice=100
            best_a=0
            best_b=0
            for j in range(0,3):
                a=copy.deepcopy(routes[index[change[j][0]]])
                b=copy.deepcopy(routes[index[change[j][1]]])
                if len(new_route[change[j][0]][i])>0:
                    t1=a.index(new_route[change[j][0]][i][0])
                    for k in range(len(new_route[change[j][0]][i])):
                        a.pop(t1)
                    a=a[:t1]+new_route[change[j][1]][i]+a[t1:]
                else:
                    a=a[:-1]+new_route[change[j][1]][i]+a[-1:]

                if len(new_route[change[j][1]][i])>0:
                    t2=b.index(new_route[change[j][1]][i][0])
                    for k in range(len(new_route[change[j][1]][i])):
                        b.pop(t2)
                    b=b[:t2]+new_route[change[j][0]][i]+b[t2:]
                else:
                    b=b[:-1]+new_route[change[j][0]][i]+b[-1:]

                if check(a) and check(b) and cost_route(a)+cost_route(b)+cost_route(routes[index[3-change[j][0]-change[j][1]]])<best:
                    best=cost_route(a)+cost_route(b)+cost_route(routes[index[3-change[j][0]-change[j][1]]])
                    best_choice=j
                    best_a=a
                    best_b=b
            if best != 100000000:
                routes[index[change[best_choice][0]]]=best_a
                routes[index[change[best_choice][1]]]=best_b
        index=0
        result={}
        a = route__1(routes)
        for x in (a):
            if x!=[1,1]:
                result[index]=x
                index+=1
        return cost2((route_1(a))), result


    def dif(i,j):
        # max_cap, xcoord, ycoord, demand, e_time, l_time, s_time, data = load_data()
        return abs(int(e_time[i])-int(e_time[j]))+abs(int(l_time[i])-int(l_time[j]))


    def ls2(t,epochs=10):
      t1=copy.deepcopy(t)
      routes=[]
        # routes = t1
      for i in range(len(t1)):
        routes.append(t1[i])
      routes=route_1(routes)
      while routes[-1]==[0,0]:
        routes.pop()
      for i in range(epochs):
        for i in range(epochs):
            choose_custom=random.sample(range(1, cus_num+1 ), 1)[0]
        for i in range(len(routes)):
            if choose_custom in routes[i]:
                choose_route=i
        best=1000
        for i in range(len(routes)):
            if i != choose_route:
                for j in range(len(routes[i])):
                    if dif(routes[i][j],choose_custom)<best:
                        best=dif(routes[i][j],choose_custom)
                        best_custom=routes[i][j]
                        best_route=i
        new_route1=copy.deepcopy(routes[best_route])
        new_route2=copy.deepcopy(routes[choose_route])
        new_route2[routes[choose_route].index(choose_custom)]=best_custom
        new_route1[routes[best_route].index(best_custom)]= choose_custom
        if check(new_route1) and check(new_route2):
            routes[best_route]=new_route1
            routes[choose_route]=new_route2
      index=0
      result={}
      a = route__1(routes)
      for x in (a):
        if x!=[1,1]:
            result[index]=x
            index+=1
      return cost2((route_1(a))), result


    # Local search Cuong 

    # #tìm hàm neighboor[i] = j <=> khách hàng thứ i gần j nhất



    #chèn 1 điểm vào 1 đường
    def insert_cus(cus1,route):


        min_insert = 10**10
        best_index1=0
        route_in = copy.deepcopy(route)
        for i in range(1,len(route)):
            route_in.insert(i,cus1)
            if check(route_in):
                if cost_route(route_in)<min_insert:
                    min_insert = cost_route(route_in)
                    best_index1  = i
            del route_in[i]
        return best_index1


    def min1_dist(ch_cus,routes): 

        neigh_routes = -1
        ch_routes = -1              #ch_cus : là khách hàng được chọn để tráo vị trí với neighboor của nó
        routes1=route_1(routes)
        while routes1[-1]==[0,0]:
            routes1.pop()
        neigh_cus = neighboor[ch_cus]      #neighboor của nó
        # print(neigh_cus)
        for i1 in range (len(routes1)):
            if ch_cus in routes1[i1]:     
                # print('Done_1')     #vị trí route chứa ch_cus
                ch_routes = i1
            if neigh_cus in routes1[i1]:  
                # print('Done_2')           #vị trí route chứa neighboor
                neigh_routes = i1    
        # print(neigh_routes, ch_routes)
        if neigh_routes == -1 and ch_routes == -1:
            return False, 0.0
        if neigh_routes == ch_routes:
            return False,0.0
        else:
            new_ch_routes = copy.deepcopy(routes1[ch_routes])
            # print(neigh_routes)
            new_neigh_routes = copy.deepcopy(routes1[neigh_routes])      
            sum1 = cost_route(new_ch_routes)+ cost_route(new_neigh_routes)
            find_vt =  insert_cus(ch_cus,new_neigh_routes)
            if find_vt>0:
                new_neigh_routes.insert(find_vt,ch_cus)  
                # if ch_cus in new_ch_routes:       # chèn ch_cus vào vị trí best của neigh
                new_ch_routes.remove(ch_cus) 
                # else:
                #     return False, 0.0 #xóa ch_cus khỏi route ban đầu của nó 
                if cost_route(new_ch_routes)+cost_route(new_neigh_routes) < sum1:        #nếu mà quãng đường tối ưu thì OK
                    routes1[ch_routes]=new_ch_routes
                    routes1[neigh_routes]=new_neigh_routes

                    return True,route__1(routes1)
                else:
                    return False,0.0
            else:
                return False,0.0


    def ls3(t):
        t1=copy.deepcopy(t)
        routes=[]
          # routes = t1
        for i in range(len(t1)):
          routes.append(t1[i])
        for choose_cus in range(1,cus_num+1):
            xx1 = min1_dist(choose_cus,routes)
            if xx1[0]==True:
                routes = xx1[1]
        index=0
        result={}
        a = routes
        for x in (a):
            if x!=[1,1]:
                result[index]=x
                index+=1
        return cost2((route_1(a))), result



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
        # from Injection import *
        from Cross_Exchange import *
        # from Local_search import * 
        from Draw import *


        max_cap, xcoord, ycoord, demand, e_time, l_time, s_time, data = load_data()
        cus_num=len(demand)-1
        print(cus_num)
        neighboor=[[] for _ in range(cus_num+1)]
        neighboor[0]=0
        for i in range(1,cus_num+1):
            min_dis = 10**10
            for j in range(1,cus_num+1):                
                if j!=i:
                    if ((xcoord[i]-xcoord[j])**2+(ycoord[i]-ycoord[j])**2)**(1/2)<min_dis:
                        min_dis = ((xcoord[i]-xcoord[j])**2+(ycoord[i]-ycoord[j])**2)**(1/2)              
                        vt = j 
            neighboor[i] = vt





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


