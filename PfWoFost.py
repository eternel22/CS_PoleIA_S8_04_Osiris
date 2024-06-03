import numpy as np
import scipy
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from dataproviders import parameters, agromanagement, weather
from pcse.models import Wofost72_WLP_FD
import copy
import os, sys
matplotlib.style.use('ggplot')

class HiddenPrints:
    ''' Classe pour empécher l'impression par défaut de python'''
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class ParticleSetWoFoSt():
    '''
    Classe abstraite pour génerer et gérer un ensemble d'instances WoFoSt. Elle prend en entrée les paramètres, le temps, et les données météorologiques.
    '''
    def __init__(self,ensemble_size = 50, parameters=parameters, weather=weather, agromanagement=agromanagement, override_parameters=None):        
        '''
        Initialiser les particules. Elles seront à l'instant 0 et couvriront les plages données dans override_parameters.

        Entrée:
        - ensemble_size: nombre de particules à créer (int)
        - parameters, weather, agromanagement: fichiers compatibles avec WoFoSt
        - override_parameters: respecter le format (NomParamètre,[valeur]*ensemble_size)
        
        Sortie:
        - Aucune. Élément "set" initialisé, contenant les paramètres.

        '''
        self.__ensemble_size = ensemble_size
        self.ensemble = []
        for i in range(ensemble_size):
            p = copy.deepcopy(parameters)
            if override_parameters == None:
                Warning("[ParticleSetWoFoSt] No override parameters given.")
            else:
                for par, distr in override_parameters.items():
                    # This is were the parameters are modified using the set_override command and the distribution given in override_parameters
                    p.set_override(par, distr[i])
            member = Wofost72_WLP_FD(p, weather, agromanagement)
            self.ensemble.append(member)
    
        override_parameters = pd.DataFrame(override_parameters)
        self.set = {self.ensemble[element]:dict(override_parameters.iloc[[element]]) for element in range(self.__ensemble_size)}
    
    def getDay(self):
        """
        Récupérer le jour actuel (commence à 0 pour le début de la simulation)
        """
        return len(self.ensemble[0].get_output("day"))
    
    def getValues(self,variable="LAI",end=True):
        """
        Récupère les valeurs de l'ensemble.
        """
        return [element.get_variable(variable) for element in self.ensemble]

    def getParameters(self, index):
        """
        Récupère les paramètres de l'élément d'index donné
        """
        return self.set[self.ensemble[index]]

class PfWoFoSt():
    
    def __init__(self, ensemble_size, parameters=parameters, weather=weather, agromanagement=agromanagement, override_parameters=None, override_ranges=None):
        """
        Initialise l'objet PfWoFoSt. Crée l'ensemble initial de particules, à l'instant t=0
        - parameters, weather, agromanagement: fichiers compatibles WoFoSt
        - override_parameters: noms des paramètres à faire varier
        - override_ranges: moyenne et écart-type de chaque paramètre à faire varier.
         parameters, weather, agromanagement: WoFoSt compatible files
        Sortie:de_parameters: names of parameters to vary
        - Aucune. Élément "set" initialisé, contenant les paramètres.
        """
        self.__ensemble_size        = ensemble_size
        self.__override_parameters  = override_parameters
        self.__override_ranges      = override_ranges
        self.date                   = None
        self._observations = {}

        self.override_parameters    = {override_parameters[index]:np.random.normal(override_ranges[index][0],override_ranges[index][1], self.__ensemble_size) for index in range(len(override_ranges))}

        particles                   = ParticleSetWoFoSt(parameters=parameters, weather=weather, agromanagement=agromanagement, ensemble_size=self.__ensemble_size, override_parameters=self.override_parameters)
        
        self.particle_set           = particles.set
        self.weights                = [1/self.__ensemble_size for element in range(self.__ensemble_size)]        
        self.log_part               = self.particle_set
    
    def get_particles_last_value(self, days=0,STATE='LAI'):
        '''
        Récupère les valeurs finales des particules. Utile pour le processus d'assimilation.
        '''
        for element in self.particle_set.keys():
            element.run(days)

        return np.array([element.get_variable(STATE) for element in self.particle_set.keys()])

    def moveForward(self, days=1,date=None):
        '''
        Avancer dans le temps: "prédire" les variables
        '''
        if date:
            for element in self.particle_set.keys():
                with HiddenPrints():
                    element.run_till(date)
                return
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

    def clean(self):
        """
        Enlever les particules qui ont des valeurs négatives (absurdes)
        """
        values = self.get_particles_last_value()
        for iV in range(len(values)):
            if values[iV]<0:
                print("[REMOVED] Removed element",list(self.particle_set.keys())[iV])
                self.particle_set.pop(list(self.particle_set.keys())[iV])
                self.weights = np.delete(self.weights,iV)

    def assimilate(self, obs):
        """
        On doit mettre à jour en deux étapes:
        - avancer à l'observation suivante
        - mettre à jour les particules, les poids en conséquence
        """
        print("Date {}, values:{}".format(obs[0],obs[1]))
        self.moveForward(date=obs[0])
        w_distance = 0
        if 'LAI' in obs[1].keys():
            curr_value_LAI  = self.get_particles_last_value(STATE='LAI')
            w_distance += scipy.stats.norm(0,0.5).pdf(np.abs(curr_value_LAI-obs[1]['LAI'][0])+ 1.e-300)
        if 'SM' in obs[1].keys():
            curr_value_SM   = self.get_particles_last_value(STATE='SM')
            w_distance += scipy.stats.norm(0,0.5).pdf(np.abs(curr_value_SM-obs[1]['SM'][0])+ 1.e-300)
        # w_distance      = (scipy.stats.norm(0,0.5).pdf(w_distance_LAI) + scipy.stats.norm(0,0.5).pdf(w_distance_SM))/2
        w_distance = w_distance/len(obs[1].keys())
        self.weights   *= w_distance
        self.weights   /= sum(self.weights)

    def displayLAIsM(self,fig=None,all=False,ax=None):
        """ 
        Afficher les valeurs d'intérêt pour les différents graphiques.
        "all" permet de tracer tous les graphiques au lieu des plus récents
        """
        if fig is None or ax is None:
            print("[Display] No fig supplied, creating a new one.")
            fig, ax = plt.subplots(2,1)
        if not all:
            for index in range(len(self.particle_set)):
                ax[0].plot(pd.DataFrame(list(self.particle_set.keys())[index].get_output())['LAI'],color='black',alpha=self.weights[index])
                ax[1].plot(pd.DataFrame(list(self.particle_set.keys())[index].get_output())['SM'],color='black',alpha=self.weights[index])
        else:
            for index in range(len(self.log_part)):
                ax[0].plot(pd.DataFrame(list(self.log_part.keys())[index].get_output())['LAI'],color='black')
                ax[0].plot(pd.DataFrame(list(self.log_part.keys())[index].get_output())['SM'],color='black')
        
    def estimate(self):
        """
        D'après les particules, calculer la moyenne et l'écart-type de notre simulation (supposée précise)
        """
        # modifié pour prendre en compte uniquement SM
        pos = self.get_particles_last_value(STATE='SM')
        mean= np.average(pos,weights=self.weights)
        var = np.average((pos - mean)**2, weights=self.weights, axis=0)
        return mean, var

    def neff(self):
        return 1. / np.sum(np.square(self.weights))

    def index_resample(self,indexes,k=5):
        """
        D'après une liste d'index, recréer de nouvelles particules proches de ces index.
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
        new_set = ParticleSetWoFoSt(self.__ensemble_size-k,override_parameters=new_override_parameters)
        self.particle_set = self.particle_set | new_set.set
        self.log_part = self.log_part | self.particle_set
        self.weights = [1/self.__ensemble_size]*self.__ensemble_size

    def get_K_top(self,k=5):
        indexes = []
        d = self.weights.copy()
        for i in range(k):
            indexes.append(d.argmax())
            d[indexes[-1]] = 0
        return indexes

    def batchAssimilate(self,obs_list):
        k=int(self.__ensemble_size/2)
        fig, ax = plt.subplots(2,1,figsize=(16,16))
        if type(obs_list)!=list:
            obs_list = [obs_list]
        for obs in obs_list:
            self._observations[obs[0]] = obs[1]
            print("\n=====[Assimilate] Currently on observation {}/{} with {} particles".format(obs_list.index(obs)+1,len(obs_list), len(self.particle_set)))
            self.assimilate(obs)
            print("[Assimilate] Updated weights. Current SM estimate: ",self.estimate())
            # ax[0].errorbar(len(pd.DataFrame(list(self.particle_set.keys())[0].get_output())['LAI']),obs[1][],yerr=obs[1]*0.1, fmt='o',color='red')
            # ax[1].errorbar(len(pd.DataFrame(list(self.particle_set.keys())[0].get_output())['LAI']),obs[2],yerr=obs[2]*0.1, fmt='o',color='red')
            if self.neff() < self.__ensemble_size/2:
                print("[Resampling] from neff",self.neff())
                self.index_resample(self.get_K_top(k),k)
                # recompute weights with new particles
                self.assimilate(obs)
                print("[Resampling] New estimate:",self.estimate())
            self.displayLAIsM(fig=fig,ax=ax)
        self.displayLAIsM(fig=fig,ax=ax)

    def avg(self,plot=False,obs_list=[],fig=None, ax=None,state=None):
        """
        Renvoie la moyenne des particules. Si plot=True, affiche le graphique.
        """
        sum1 = pd.DataFrame()
        sum2 = pd.DataFrame()
        sum3 = pd.DataFrame()
        if fig is None and plot == True:
            fig, ax = plt.subplots(2,1)
            pass
        for element in self.particle_set:
            sum1[element] = pd.DataFrame(element.get_output())['LAI']
            sum2[element] = pd.DataFrame(element.get_output())['SM']
            if state != None:
                sum3[element] = pd.DataFrame(element.get_output())[state]
        if not plot:
            if state != None:
                return sum1.mean(axis=1), sum2.mean(axis=1), sum3.mean(axis=1)
            return sum1.mean(axis=1), sum2.mean(axis=1)
        x = list(pd.DataFrame(element.get_output())['day'])
        ax[0].plot(sum1.mean(axis=1),color='black')
        ax[1].plot(sum2.mean(axis=1),color='black')
        for obs in obs_list:
            ax[0].errorbar((obs[0] - list(self.particle_set.keys())[0].get_output()[0]['day']).days,obs[1],yerr=obs[1]*0.1, fmt='o',color='red')
            ax[1].errorbar((obs[0] - list(self.particle_set.keys())[0].get_output()[0]['day']).days,obs[2],yerr=obs[2]*0.2, fmt='o',color='red')
    
    def get_current_date(self):
        return list(self.particle_set.keys())[0].get_variable("day")

    def completeSim(self):
        for element in self.particle_set:
            element.run_till_terminate()
    
    def getState(self, specific="all"):
        specifics = {"all": [self.__parameters,self.__weather, self.__agromanagement]}
        return specifics[specific]
    
    def evaluate(self):

        def diff(a,b):
            t = 0
            for el in set(a.keys()).intersection(b.keys()):
                t+= (a[el]-b[el])**2
            return np.root(t/len(set(a.keys().intersection(b.keys())))) # RMSE
        
        globalv = {}
        return diff(pd.DataFrame(self.avg()[0]),pd.DataFrame({element:{'LAI':self._observations[element]['LAI'][0], 'SM':self._observations[element]['SM'][0]} for element in self._observations.keys()}).transpose()['LAI'])
    
    def get_df(self):
        results = pd.DataFrame({'day':pd.DataFrame(list(self.particle_set.keys())[0].get_output())['day'],
                                'SM':pd.DataFrame(self.avg()).transpose()[1],
                                'TAGP':pd.DataFrame(self.avg(state='TAGP')).transpose()[2]})
        results.set_index('day',inplace=True)
        return results