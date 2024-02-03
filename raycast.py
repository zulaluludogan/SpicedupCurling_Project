import numpy as np
import utility 

class RayCast:

    def __init__(self, target: np.ndarray, obstacles: np.ndarray, 
                 enemy_pucks: np.ndarray = [], our_pucks: np.ndarray = []) -> None:
         
        self._target = target
        self._obstacles = obstacles
        self._enemy_pucks = enemy_pucks
        self._our_pucks = our_pucks

    @property
    def target(self):
        return self._target # just center coordinates no radius

    @property
    def obstacles(self):
        return self._obstacles

    @property
    def enemy_pucks(self):
        return self._enemy_pucks # just center coordinates no radius
    
    @property
    def our_pucks(self):
        return self._our_pucks # just center coordinates no radius
    
    @target.setter
    def target(self, t):
        self._target = t

    @obstacles.setter
    def obstacles(self, o):
        self._obstacles = o

    @enemy_pucks.setter
    def enemy_pucks(self, ep):
        self._enemy_pucks = ep

    @our_pucks.setter
    def our_pucks(self, op):
        self._our_pucks = op

    
    # Intersection Functions
    def intersect_circle(self, start_point: np.ndarray, angle: float, center: np.ndarray, radius: float) -> np.ndarray:
        '''
        checks if there is intersection with the target goal

        returns: -1 -> no intersection, np.ndarray -> intersection point x, intersection point y, distance
        '''

        # https://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm/86428#86428

        direction_vector = np.array((np.cos(angle), np.sin(angle)))

        a = np.dot(direction_vector, direction_vector)
        b = 2 * np.dot(direction_vector, start_point - center)
        c = np.dot(start_point, start_point) + np.dot(center, center) - 2 * np.dot(start_point, center) - radius ** 2

        disc = b**2-4*a*c

        epsilon = 0.01

        if disc < epsilon:
            return [0,0,-1]

        sqrt_disc = np.sqrt(disc)
        t1 = (-b+sqrt_disc)/(2*a)
        t2 = (-b-sqrt_disc)/(2*a) 

        intersection_point1 = start_point + t1 * direction_vector
        intersection_point2 = start_point + t2 * direction_vector 
        closest_intersection_point = intersection_point1  if abs(t1)<abs(t2) else intersection_point2
        mid_distance = np.linalg.norm(intersection_point1-intersection_point2)/2
        
        return np.array([closest_intersection_point[0], closest_intersection_point[1], min(abs(t1), abs(t2)) + mid_distance])

    def intersect_line(self, start_point: np.ndarray, angle: float, boundary_point1: np.ndarray, boundary_point2: np.ndarray) -> np.ndarray:
        '''
        checks if there is intersection with the obtacles and boundaries

        returns: -1 -> no intersection, np.ndarray -> intersection point x, intersection point y, distance
        '''

        # https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect

        direction_vector_puck = np.array((np.cos(angle), np.sin(angle)))
        direction_vector_obstacle = boundary_point2-boundary_point1
        direction_vectors_crossed_mag = utility.cross_product_2d(direction_vector_puck, direction_vector_obstacle)

        # print(direction_vector_puck)
        # print(direction_vector_obstacle)
        # print(direction_vectors_crossed_mag)

        t = utility.cross_product_2d((boundary_point1 - start_point), direction_vector_obstacle) / direction_vectors_crossed_mag
        u = utility.cross_product_2d((boundary_point1 - start_point), direction_vector_puck)     / direction_vectors_crossed_mag

        # print(t)
        # print(u)

        intersection_point = start_point + t * direction_vector_puck

        epsilon = 0.01

        if direction_vectors_crossed_mag == 0 and utility.cross_product_2d(boundary_point1-start_point, direction_vector_puck) == 0: 
            return np.array([boundary_point1[0], boundary_point2[1], np.linalg.norm(boundary_point1-start_point)])
        elif direction_vectors_crossed_mag != 0 and (-epsilon<=u<=1+epsilon):
            return np.array([intersection_point[0], intersection_point[1], abs(t)])
        else: return [0,0,-1]

    def get_all_collisions(self, start_point: np.ndarray, angle: float, initial_distance: float) -> list:

        # Array to keep track of all possible intersections
        all_intersections = []
        
        R_TARGET = 130
        R_PUCK = 30

        # Check Circle Intersection
        target_intersection = self.intersect_circle(start_point, angle, np.array(self.target[0:2]), R_TARGET)
        if target_intersection[2] != -1: 
            target_intersection[2] += initial_distance
            all_intersections.append(['t', target_intersection])

        # Check Obstacles and Boundary Intersections
        for i in range(len(self.obstacles)):
            obstacle_intersection = self.intersect_line(start_point, angle,
                                                         np.array([obstacles[i][0],obstacles[i][1]]),
                                                         np.array([obstacles[i][2],obstacles[i][3]])
                         )

            if obstacle_intersection[2] != -1:
                obstacle_intersection[2] += initial_distance
                all_intersections.append([f'b{i}', obstacle_intersection])
        
        # Check Enemy Puck Intersections
        for i in range(len(self.enemy_pucks)):
            puck_intersection = self.intersect_circle(start_point, angle, 
                                                      np.array([our_pucks[i][0], our_pucks[i][1]]),
                                                      R_PUCK)
           
            if puck_intersection[2] != -1:
                puck_intersection[2] += initial_distance
                all_intersections.append([f'e{i}', puck_intersection])
        
        # Check Our Puck Intersections
        for i in range(len(self.our_pucks)):
            puck_intersection = self.intersect_circle(start_point, angle, np.array([our_pucks[i][0], our_pucks[i][1]]), R_PUCK)  

            if puck_intersection[2] != -1:
                puck_intersection[2] += initial_distance
                all_intersections.append([f'o{i}', puck_intersection])
        
        all_intersections = utility.sort_by_distance2(all_intersections)
        
        epsilon = 0.01
        if all_intersections != []:
            
            while True:
                if abs(all_intersections[0][1][2]) <= initial_distance + epsilon:
                    del all_intersections[i]
                
                if all_intersections == []: break
                
                if abs(all_intersections[0][1][2])> initial_distance + epsilon: break
            return all_intersections
    
    def my_stupid_search(self, start_point, angle, cost_so_far):
        
        came_from = [start_point]

        start_point_collisions = self.get_all_collisions(start_point, angle, 0)
        
        if start_point_collisions == [] or start_point_collisions == None: return [], -1, 0
        current = start_point_collisions.pop(0)
        num_collisions = 0

        for _ in range(0,2):
            
            if current[0][0] == 't':
                came_from.append((current[1][0], current[1][1]))
                return came_from, current[1][2], num_collisions

            elif current[0][0] == 'o' or current[0][0] == 'e':
                return [], [0,0,-1], 0

            came_from.append((current[1][0], current[1][1]))
            
            all_collisions= self.get_all_collisions(np.array([current[1][0], current[1][1]]), utility.reflect(angle), current[1][2] + cost_so_far)
            
            if all_collisions == []: return [],-1, 0
            
            current = all_collisions.pop(0)
            num_collisions += 1
               
    def ray_cast(self, start_point) -> tuple:

        angles = np.linspace(0, np.pi, 180)
        
        possible_paths = []

        for angle in angles:
            x = self.my_stupid_search(start_point, angle, 0)
            if x[1] != -1: 
                possible_paths.append(x)
        
        
        
        return utility.sort_by_distance3(possible_paths)[0]
        # for i in range(len(possible_paths)):
            # if possible_paths[i][1] != -1:
                # return (possible_paths[i])
        
target_goal = np.array([3,6,1])
obstacles = np.array([(-5,-10,-5,10),])
enemy_pucks = np.array([])
our_pucks = np.array([])

if __name__ == "__main__":

    start_point = np.array([0,0])
    angle = 2.76

    raycaster = RayCast(target_goal, obstacles, enemy_pucks, our_pucks)
    

    #print(intersect_line(start_point, angle, boundary_point1, boundary_point2))
    
    old_distance = 0
    
    #print(raycaster.get_all_collisions(start_point, angle, old_distance))    
    #print(raycaster.raycast_single(start_point, angle, old_distance, 0))
    #print(raycaster.intersect_circle(np.array([0,0]), np.pi/2 + 0.3, np.array([0,7]), 3))
    #print(raycaster.my_stupid_search(start_point, 0.0, 0))
    #print(raycaster.intersect_circle(np.array([-5, 2]), 0.38, np.array([3,6]), 1))
    #print(raycaster.get_all_collisions(np.array([-5,2]), 0.38, 0))
    print(raycaster.ray_cast(start_point, angle))