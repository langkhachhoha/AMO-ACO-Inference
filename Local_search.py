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



def check(route):
    # max_cap, xcoord, ycoord, demand, e_time, l_time, s_time, data = data_zip
    max_cap, xcoord, ycoord, demand, e_time, l_time, s_time, data = load_data()
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


