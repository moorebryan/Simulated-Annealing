# Import modules
import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt
import random
import math


# we will now tailor the above algorithm to solve the travelling salesman problem
class Simm_Anneal(object):
    def __init__(self, coords, maxsteps, debug=True):
        #self.current_state = initial_state
        self.maxiter = maxsteps
        self.debug = debug
        self.coords = coords
        self.N = len(coords)
        self.best_solution = None
        self.best_fitness = float("Inf")
        self.states, self.costs=[],[]
        self.nodes = [i for i in range(self.N)]
        self.starting_solution = self.initial_solution
        self.current_state,self.current_cost = self.starting_solution
        self.fitness_list = []
    
    #find euclidean distance between the nodes
    def dist(self, node_0, node_1):
        """
        Euclidean distance between two nodes.
        """
        coord_0, coord_1 = self.coords[node_0], self.coords[node_1]
        return math.sqrt((coord_0[0] - coord_1[0]) ** 2 + (coord_0[1] - coord_1[1]) ** 2)
    
    @property
    def initial_solution(self):
        """
        Find the best nearest neighbor.
        """
        cur_node = random.choice(self.nodes)  # start from a random node
        solution = [cur_node]
        free_nodes = set(self.nodes)
        free_nodes.remove(cur_node)
        while free_nodes:
            next_node = min(free_nodes, key=lambda x: self.dist(cur_node, x))  # nearest neighbour
            free_nodes.remove(next_node)
            solution.append(next_node)
            cur_node = next_node

        cur_fit = self.fitness(solution)
        if cur_fit < self.best_fitness:  # If best found so far, update best fitness
            self.best_fitness = cur_fit
            self.best_solution = solution
        self.states.append(solution)
        self.costs.append(cur_fit)
        return solution, cur_fit
    
    @staticmethod
    def temperature(x):
        '''alter temperature as a function of iteration step'''
        return max(0.01, min(1, 1 - x))
    
    @staticmethod
    def acceptance_probability(cost, new_cost, temperature):
        cost_diff = new_cost-cost
        if cost_diff < 0:
            return 1
        else:
            p = np.exp(- (cost_diff) / temperature)
            return p
    
    def fitness(self, solution):
        """
        Total distance of the current solution path.
        """
        cur_fit = 0
        for i in range(self.N):
            cur_fit += self.dist(solution[i % self.N], solution[(i + 1) % self.N])
        return cur_fit

    def anneal(self):
        '''
        Execute simulated annealing
        '''
        
        for step in range(self.maxiter):
            self.fraction = step / float(self.maxiter)
            T = self.temperature(self.fraction)
            new_state=[item for item in self.current_state]
            l = random.randint(2, self.N - 1)
            i = random.randint(0, self.N - l)
            new_state[i : (i + l)] = reversed(new_state[i : (i + l)])
            new_cost = self.fitness(new_state)
            if self.debug:
                print("Step #{:>2}/{:>2} : T = {:>4.3g}, state = {:>4.3g}, cost = {:>4.3g}, new_state = {:>4.3g}, new_cost = {:>4.3g} ...".format(step, maxsteps, T, self.current_state, self.current_cost, new_state, new_cost))
            if self.acceptance_probability(self.current_cost, new_cost, T) > rn.random():
                self.current_state, self.current_cost = new_state, new_cost
                self.states.append(self.current_state)
                self.costs.append(self.current_cost)
                self.fitness_list.append(new_cost)
        return self.states, self.costs

    def plotTSP(self, paths, points, title):
        """
        path: List of lists with the different orders in which the nodes are visited
        points: coordinates for the different nodes
        num_iters: number of paths that are in the path list
        """
        # Unpack the primary TSP path and transform it into a list of ordered
        # coordinates
        x = []; y = []
        for i in paths[0]:
            x.append(points[i][0])
            y.append(points[i][1])
        plt.figure(figsize=(10,7))
        plt.plot(x, y, 'co')
        # Set a scale for the arrow heads (there should be a reasonable default for this, WTF?)
        a_scale = float(max(x))/float(100)
        # Draw the primary path for the TSP problem
        plt.arrow(x[-1], y[-1], (x[0] - x[-1]), (y[0] - y[-1]), head_width = a_scale,
                color ='g', length_includes_head=True)
        for i in range(0,len(x)-1):
            plt.arrow(x[i], y[i], (x[i+1] - x[i]), (y[i+1] - y[i]), head_width = a_scale,
                    color = 'g', length_includes_head = True)
        #Set axis too slitghtly larger than the set of x and y
        plt.xlim(min(x)*1.1, max(x)*1.1)
        plt.ylim(min(y)*1.1, max(y)*1.1)
        plt.title(title)
        plt.xlabel('x-coordinates')
        plt.ylabel('y-coordinates')
        plt.show()
    
    def visualize_routes(self):
        """
        Visualize the TSP route with matplotlib.
        """
        self.plotTSP([self.states[0]], self.coords, title='Initial starting arrangement (connecting nearest nodes)')
        self.plotTSP([self.states[-1]], self.coords, title='Final arrangement (after simulated annealing optimization)')

    def plot_learning(self):
        """
        Plot the fitness through iterations.
        """
        plt.figure(figsize=(10,7))
        plt.plot([i for i in range(len(self.fitness_list))], self.fitness_list)
        plt.ylabel("Fitness")
        plt.xlabel("Iteration")
        plt.title('Cost function over iterations')
        plt.show()
