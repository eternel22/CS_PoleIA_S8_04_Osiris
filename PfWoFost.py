from data_gen import DataGenerator
from kalmanWoFost import KalmanWofostDA
import numpy as np
import scipy
import datetime as dt
import pandas as pd
from matplotlib import pyplot as plt
from dataproviders import parameters, agromanagement, weather
from pcse.models import Wofost72_WLP_FD
import copy
import os, sys

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class ParticleSetWoFost():
    def __init__(self,ensemble_size = 50, parameters=parameters, weather=weather, agromanagement=agromanagement, override_parameters=None):        
        '''
        Initialize particles. These will be at time 0 and cover the ranges given in the override_parameters.

        Input:
        - ensemble_size: number of particles to create (int)
        - parameters, weather, agromanagement: WoFost compatible files
        - override_parameters: respect the format (ParameterName,[value]*ensemble_size)
        
        Output:
        - None. Initialised "set" element, containing the parameters.

        '''
        self.__ensemble_size = ensemble_size
        self.ensemble = []
        for i in range(ensemble_size):
            p = copy.deepcopy(parameters)
            if override_parameters == None:
                Warning("[ParticleSetWoFost] No override parameters given.")
            else:
                for par, distr in override_parameters.items():
                    # This is were the parameters are modified using the set_override command
                    p.set_override(par, distr[i])
            member = Wofost72_WLP_FD(p, weather, agromanagement)
            self.ensemble.append(member)
    
        override_parameters = pd.DataFrame(override_parameters)
        self.set = {self.ensemble[element]:dict(override_parameters.iloc[[element]]) for element in range(self.__ensemble_size)}
    
    def getDay(self):
        """
        Get currentday (starts at 0 for simulation beginning)
        """
        return len(self.ensemble[0].get_output("day"))
    
    def getValues(self,variable="LAI",end=True):
        """
        Get values of the set.
        """
        return [element.get_variable(variable) for element in self.ensemble]

    def getParameters(self, index):
        """
        Return the parameters of a given index element
        """
        return self.set[self.ensemble[index]]

class PfWoFost():
    def __init__(self, ensemble_size, parameters=parameters, weather=weather, agromanagement=agromanagement, override_parameters=None, override_ranges=None):
        """
        Initialise the PfWofost object. Creates the initial set of particles, at time t=0

        Input:
        - ensemble_size: number of particles to create (int)
        - parameters, weather, agromanagement: WoFost compatible files
        - override_parameters: names of parameters to vary
        - override_ranges: mean and std-var of each parameter to vary.
        
        Output:
        - None. Initialised "set" element, containing the parameters.
        """
        self.__ensemble_size        = ensemble_size
        self.__override_parameters  = override_parameters
        self.__override_ranges      = override_ranges
        self.date                   = None

        self.override_parameters    = {override_parameters[index]:np.random.normal(override_ranges[index][0],override_ranges[index][1], self.__ensemble_size) for index in range(len(override_ranges))}

        particles                   = ParticleSetWoFost(parameters=parameters, weather=weather, agromanagement=agromanagement, ensemble_size=self.__ensemble_size, override_parameters=self.override_parameters)
        
        self.particle_set           = particles.set
        self.weights                = [1/self.__ensemble_size for element in range(self.__ensemble_size)]        
        self.log_part               = self.particle_set
    
    def get_particles_last_value(self, days=0,STATE='LAI'):
        '''
        Fetch the final values of the particles. This is useful for the assimilation process.
        '''
        for element in self.particle_set.keys():
            element.run(days)

        return np.array([element.get_variable(STATE) for element in self.particle_set.keys()])

    def predict(self, days=1,date=None):
        '''
        Move forward in time: "predict" the variables
        '''
        if date:
            for element in self.particle_set.keys():
                with HiddenPrints():
                    element.run_till(date)
        else:
            for element in self.particle_set.keys():
                element.run(days)

        # Let's clean the ensemble: we will drop elements that are negative
        values = self.get_particles_last_value()
        for iV in range(len(values)):
            if values[iV]<0:
                print("[REMOVED] Removed element",list(self.particle_set.keys())[iV])
                self.particle_set.pop(list(self.particle_set.keys())[iV])
                self.weights = np.delete(self.weights,iV)

    def update(self, obs):
        """
        We need to do this in two steps:
        - move forward to next observation
        - update particles, self.weights accordingly
        """
        self.predict(date=obs[0])
        curr_value      = self.get_particles_last_value()
        w_distance      = np.abs(curr_value-obs[1])+ 1.e-300
        w_distance      = scipy.stats.norm(0,0.5).pdf(w_distance)
        self.weights   *= w_distance
        self.weights   /=  sum(self.weights)

    def plot_p(self,fig=plt,all=False):
        """ 
        Display the values of interest for the different graphs.
        "all" allows to trace all the plots instead of just the most recent ones
        """
        if not all:
            for index in range(len(self.particle_set)):
                fig.plot(pd.DataFrame(list(self.particle_set.keys())[index].get_output())['LAI'],color='black',alpha=self.weights[index])
        else:
            for index in range(len(self.log_part)):
                fig.plot(pd.DataFrame(list(self.log_part.keys())[index].get_output())['LAI'],color='black')
        
    def estimate(self):
        """
        Based on the particles, compute the weighted average and std of our simulation (supposedly accurate)
        """
        pos = self.get_particles_last_value()
        mean= np.average(pos,weights=self.weights)
        var = np.average((pos - mean)**2, weights=self.weights, axis=0)
        return mean, var

    def neff(self):
        return 1. / np.sum(np.square(self.weights))

    def index_resample(self,indexes):
        """
        From a given list of index, recreate new particles close to these indexes.
        """
        keys = list(self.particle_set.keys())
        # Keep the top 5 particles and their weights
        self.particle_set = {keys[index]:self.particle_set[keys[index]] for index in indexes}
        top_weights = [self.weights[index] for index in indexes] 
        top_weights /= sum(top_weights)
        # we have the core particles. Let's add some more:
        # First, gather information on top particles:
        ranges = []

        for name in self.__override_parameters:
            temp_l = [float(self.particle_set[keys[index]][name]) for index in indexes]
            mean_v = np.average(temp_l,weights=top_weights)
            std_v = min(np.average((temp_l - mean_v)**2, weights=top_weights)/10, self.__override_ranges[self.__override_parameters.index(name)][1])
            ranges.append((mean_v,std_v))
        print("[Resampled] New ranges: ", ranges)
        new_override_parameters = {self.__override_parameters[index]:np.random.normal(ranges[index][0],ranges[index][1], self.__ensemble_size) for index in range(len(ranges))}
        
        # Now, let's add some more particles:
        new_set = ParticleSetWoFost(self.__ensemble_size-5,override_parameters=new_override_parameters)
        self.particle_set = self.particle_set | new_set.set
        self.log_part = self.log_part | self.particle_set
        self.weights = [1/self.__ensemble_size]*self.__ensemble_size
    
    def get_five_top(self):
        indexes = []
        d = self.weights.copy()
        for i in range(5):
            indexes.append(d.argmax())
            d[indexes[-1]] = 0
        return indexes
    
    def assimilate(self,obs_list):
        
        if type(obs_list)!=list:
            obs_list = [obs_list]
        for obs in obs_list:
            print("===[Assimilate] Currently on observation {}/{} with {} particles".format(obs_list.index(obs)+1,len(obs_list), len(self.particle_set)))
            self.update(obs)
            print("[Assimilate] Updated weights. Current LAI estimate: ",self.estimate())
            plt.scatter(len(pd.DataFrame(list(self.particle_set.keys())[0].get_output())['LAI']),obs[1])
            if self.neff() < self.__ensemble_size/2:
                print("[Resampling] from neff",self.neff())
                self.index_resample(self.get_five_top())
                # recompute weights with new particles
                self.update(obs)
                print("[Resampling] New estimate:",self.estimate())
            self.plot_p()

        