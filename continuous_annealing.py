# Import all requisite modules
from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rn
import math


## Declare all abstract classes following this Python-3 example
class Abstract(metaclass=ABCMeta):
    @abstractmethod
    def move(self):
        """Create a state change"""
        pass


class Simm_Anneal(Abstract):
    def __init__(self, maxsteps, cost_function, interval, debug=True):
        super().__init__()
        self.maxiter=maxsteps
        self.debug = debug
        self.cost_function = cost_function
        self.interval = interval

    def round_figures(self, x, n):
        """Returns x rounded to n significant figures."""
        return round(x, int(n - math.ceil(math.log10(abs(x)))))

    def time_string(self, seconds):
        """Returns time in seconds as a string formatted HHHH:MM:SS."""
        s = int(round(seconds))  # round to nearest second
        h, s = divmod(s, 3600)   # get hours and remainder
        m, s = divmod(s, 60)     # split remainder into minutes and seconds
        return '%4i:%02i:%02i' % (h, m, s)

    def clip(self, x):
        """ Force x to be in the interval."""
        a, b = self.interval
        return max(min(x, b), a)

    def random_start(self):
        """ Random point in the interval."""
        a, b = self.interval
        return a + (b - a) * rn.random_sample()
    
    @staticmethod
    def temperature(x):
        '''alter temperature as a function of iteration step'''
        return max(0.01, min(1, 1 - x))
    
    @staticmethod
    def acceptance_probability(cost, new_cost, temperature):
        cost_diff = new_cost - cost
        if cost_diff < 0:
            # If new cost is less than old cost accept with probability of 1
            return 1
        else:
            # If new cost not better than old cost accept with probability as calculated below
            p = np.exp(- (cost_diff) / temperature)
            return p
    
    def move(self):
        '''create a state change to the original state'''
        amplitude = (max(self.interval) - min(self.interval)) * self.fraction / 10
        delta = (-amplitude/2.) + amplitude * rn.random_sample()
        return delta

    def anneal(self):
        # For this example randomize the starting place
        state = self.random_start()
        cost = self.cost_function(state)
        states, costs = [state], [cost]
        for step in range(self.maxiter):
            # Note this always adjusts temperature by 1/maxsteps
            self.fraction = step / float(self.maxiter)
            T = self.temperature(self.fraction)
            new_state = self.clip(self.move()+state)
            new_cost = self.cost_function(new_state)
            if self.debug:
                print("Step #{:>2}/{:>2} : T = {:>4.3g}, state = {:>4.3g}, cost = {:>4.3g}, new_state = {:>4.3g}, new_cost = {:>4.3g} ...".format(step, maxsteps, T, state, cost, new_state, new_cost))
            if self.acceptance_probability(cost, new_cost, T) > rn.random():
                state, cost = new_state, new_cost
                states.append(state)
                costs.append(cost)
        return state, self.cost_function(state), states, costs
