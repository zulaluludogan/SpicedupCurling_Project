import numpy as np

# Utility Functions
def reflect(angle: np.float64) -> np.float64:
    return np.float64(np.pi - angle)

def normalize(non_unit_vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(non_unit_vector)
    if norm == 0: return non_unit_vector
    return non_unit_vector / norm

def sort_by_distance(list2sort: np.ndarray) -> np.ndarray:
    return np.array(sorted(list(map(list,list2sort)), key = lambda pair: pair[1]))

def sort_by_distance2(list2sort: np.ndarray) -> np.ndarray:
    return sorted(list(map(list,list2sort)), key = lambda pair: pair[1][2])

def sort_by_distance3(list2sort: np.ndarray) -> np.ndarray:
    return sorted(list2sort, key = lambda triplet: triplet[1])

def cross_product_2d(vector1: np.ndarray, vector2: np.ndarray) -> np.float64:
    return np.float64(vector1[0] * vector2[1] - vector1[1] * vector2[0])

def make_compatible(obstacles):
    boundary_1 = (0,55,1110,55)
    boundary_2 = (0,625,1110,625)
    if obstacles == []:
        return np.array([boundary_1, boundary_2])
    
    y = [boundary_1, boundary_2]
    for obstacle in obstacles:
        y.append( (obstacle[0][0], obstacle[0][1], obstacle[1][0], obstacle[1][1]) )
        y.append( (obstacle[1][0], obstacle[1][1], obstacle[2][0], obstacle[2][1]) )
        y.append( (obstacle[2][0], obstacle[2][1], obstacle[3][0], obstacle[3][1]) )
        y.append( (obstacle[3][0], obstacle[3][1], obstacle[0][0], obstacle[0][1]) )
    
    return np.array(y)