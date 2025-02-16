from Load_data import *
from Model.Config import *
import torch 
from Model.Gen_CVRPTW_data import *

device = None 

def normalize_data():
    max_cap, xcoord, ycoord, demand, e_time, l_time, s_time, data = load_data()
    data = torch.tensor([[float(x) for x in y] for y in data])
    tsp_coordinates = data[:, 1:3] 
    demands = torch.tensor(demand, dtype = torch.float32)
    time_window = data[:, 4:]
    durations = time_window[:, -1] 
    distances = gen_distance_matrix(tsp_coordinates, device = device)


    pyg_data_normalize = gen_pyg_data_normalize(demands, time_window, durations, distances, device, scale = 1000.0)
    return pyg_data_normalize

