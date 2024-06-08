import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import pcse
from pcse.models import Wofost72_WLP_FD
from pcse.base import ParameterProvider
from pcse.db import NASAPowerWeatherDataProvider
from pcse.fileinput import YAMLAgroManagementReader, YAMLCropDataProvider
from pcse.util import WOFOST72SiteDataProvider, DummySoilDataProvider
from dataproviders import parameters, agromanagement, weather
import numpy as np
import pandas as pd


class DataGenerator():
    def __init__(self, parameters=parameters, weather=weather,agromanagement=agromanagement, randomness=1) :
        """
        Initializes an instance of the data_gen class.

        Parameters:
        - parameters (object): The parameters object (contains soil, crop and site data.) (PatameterProvider class).
        - weather (object): The weather object.
        - agromanagement (object): The agromanagement object.
        - randomness (int): The randomness value.

        Returns:
            None
        """
        self.__parameters = parameters
        self.__weather = weather
        self.__agromanagement = agromanagement
        self.__randomness = randomness
        self.model = Wofost72_WLP_FD(self.__parameters, self.__weather, self.__agromanagement)   
        self.model.run_till_terminate()
        self.output = pd.DataFrame(self.get('output')).set_index("day")
        self.max_time = len(self.output['LAI'])

    def edit(self,values={}):
        """
        Edit the parameters of the current data_gen instance
        
        Parameters
        - values dictionary:
            - contains 'r': randomnessparameter (int)
            - contains 'p': parameters (dataproviders)
            - contains 'w': weather (dataproviders)
            - contains 'a': agromanagement (dataproviders)
        
        Returns:
        - None
        """
        if 'r' in values:
            self.__randomness = values['r']
            if len(values.keys) == 1: return
        if 'p' in values:
            self.__parameters = values['p']
        if 'w' in values:
            self.__weather = values['w']
        if 'a' in values:
            self.__agromanagement = values['a']
        
        self.model = Wofost72_WLP_FD(self.__parameters, self.__weather, self.__agromanagement)   
        self.model.run_till_terminate()
        self.output = pd.DataFrame(self.get('output')).set_index("day")        
        self.max_time = len(self.output['LAI'])
        print('parameters edited in {}'.format(list(values.keys())))
    
    def get(self, what):
        """
        Fetch an item from the object

        Input:
        - what: string in ['weather','parameters','agromanagement','randomness','time','output']

        Output:
        - corresponding value

        
        """
        if what == 'output':
            return self.model.get_output()
        if what == 'weather':
            return self.__weather
        if what == 'parameters':
            return self.__parameters
        if what == 'agromanagement':
            return self.__agromanagement
        if what == 'randomness':
            return self.__randomness
        if what == 'time':
            return self.time
        
    
    def generateDataTimeSeries(self,series='LAI',interval=None,nb_points=None): 
        """
        Generate Simple TimeSeries
        
        Input:
        - series: type of data we wish to get
        - interval: interval between two points / nb_points: number of points we wish to generate (either one is sufficient)

        Output:
        - time series

        """
        interval,nb_points = self.intNb(interval,nb_points)
        ts = self.output[series][::interval]+(np.random.rand(len(self.output[series][::interval]))-1/2)*self.__randomness
        return ts  
    
    def plot(self,series='LAI',interval=None,nb_points=None):
        """
        Plot the time series

        Input:
        - series: type of data we wish to get
        - interval: interval between two points / nb_points: number of points we wish to generate (either one is sufficient)

        Output:
        - None
        """
        interval,nb_points = self.intNb(interval,nb_points)
        plt.plot(list(self.output[series]),label='original')
        plt.scatter(range(1,len(self.output[series]),interval),self.generateDataTimeSeries(series,interval,nb_points),label='generated',color='red')
        plt.legend()
        plt.show()
    
    def intNb(self,interval,nb_points):
        """
        Utility function to go from interval/nb_points to nb_points/interval (including if both are None)
        
        Input:
        - interval
        - nb_point 

        Output:
        - interval
        - nb_point 
        """
        if interval is None and nb_points is None:
            interval = 1
        elif interval is None:
            interval = int(self.max_time/nb_points)+1
        return interval,nb_points
    
    def generateLargeDataTimeSeries(self,nbOfIter=100,parameterSets=None,interval=None,nb_points=None,series='LAI'):
        """
        Generates a large dataset of time series data.

        Parameters:
        - nbOfIter (optional): The number of iterations to generate the data. Default value is 100.
        - parameterSets (optional): A list of parameter sets. Each parameter set is a dictionary containing the parameters for data generation. If not provided, a default parameter set is used.
        - interval (optional): The interval between data points. If not provided, it is calculated based on the maximum time and number of points.
        - nb_points (optional): The number of data points to generate. If not provided, it is calculated based on the maximum time and interval.
        - series (optional): The type of time series data to generate. Default value is 'LAI'.
        
        Return Value:
        - A pandas dataframe containing the generated time series data, indexed by i(Iteration),p(ParameterSetNumber).
        
        """
        interval,nb_points = self.intNb(interval,nb_points)
        TotalOutput = pd.DataFrame()
        if parameterSets is None:
            parameterSets = [{'p':self.__parameters}]
        for IparameterSet in range(len(parameterSets)):
            self.edit(parameterSets[IparameterSet])
            for iI in range(nbOfIter):
                TotalOutput['i{},p{}'.format(iI,IparameterSet)] = self.generateDataTimeSeries(series='LAI',interval=interval,nb_points=nb_points)
        return TotalOutput
            