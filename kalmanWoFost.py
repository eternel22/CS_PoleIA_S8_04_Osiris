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
import copy


class KalmanWofostDA():

    def __init__(self, parameters, weather,agromanagement, ensemble_size, override_parameters=None):
        self.__parameters = parameters
        self.__agromanagement = agromanagement
        self.__weather = weather
        
        self.__relative_day = 0   
        self.__ensemble_size = ensemble_size
        self._observations = {}
        self.ensemble = []
        for i in range(ensemble_size):
            p = copy.deepcopy(parameters)
            if override_parameters == None:
                Warning("[KalmanWoFoStDA] No override parameters given.")
            for par, distr in override_parameters.items():
                # This is were the parameters are modified using the set_override command
                p.set_override(par, distr[i])
            member = Wofost72_WLP_FD(p, weather, agromanagement)
            self.ensemble.append(member)
    
    def assimilate(self, obs):
        self._observations[obs[0]] = obs[1]
        print("[KalmanWoFoStDA] Assimilating data for {} on day {} ".format(str(obs[1]), obs[0]))
        variables_for_DA = obs[1].keys()
        collected_states = []
        for member in self.ensemble:
            member.run_till(obs[0])
            t = {}
            for state in variables_for_DA:
                t[state] = member.get_variable(state)
            collected_states.append(t)
        df_A = pd.DataFrame(collected_states)
        A = np.matrix(df_A).T
        P_e = np.matrix(df_A.cov())

        perturbed_obs = []
        for state in variables_for_DA:
            (value, std) = obs[1][state] # both are empiric values
            d = np.random.normal(value, std, (len(self.ensemble))) # perturb the observation
            perturbed_obs.append(d)
        df_perturbed_obs = pd.DataFrame(perturbed_obs).T
        df_perturbed_obs.columns = variables_for_DA
        D = np.matrix(df_perturbed_obs).T
        R_e = np.matrix(df_perturbed_obs.cov())

        # Here we compute the Kalman gain
        H = np.identity(len(obs))
        K1 = P_e * (H.T)
        K2 = (H * P_e) * H.T
        K = K1 * ((K2 + R_e).I)

        # Here we compute the analysed states
        Aa = A + K * (D - (H * A))
        df_Aa = pd.DataFrame(Aa.T, columns=variables_for_DA)
        df_Aa.head()

        for member, new_states in zip(self.ensemble, df_Aa.itertuples()):
            _ = member.set_variable("LAI", new_states.LAI)
            _ = member.set_variable("SM", new_states.SM)

    def displayLAIsM(self):
        print("[KalmanWoFoStDA] Displayint data for {} up to day {} ".format(len(self.ensemble), self.ensemble[0].get_variable("day")))
        results = [pd.DataFrame(member.get_output()).set_index("day") for member in self.ensemble]
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16,16), sharex=True)
        for member_df in results:
            member_df["LAI"].plot(style="k:", ax=axes[0])
            member_df["SM"].plot(style="k:", ax=axes[1])
        # axes[0].errorbar(self._observations.keys(), self._observations.values, fmt="o")
        # axes[1].errorbar(self._observations.keys, self._observations.values, yerr=std_sm, fmt="o")
        axes[0].set_title("Leaf area index")
        axes[1].set_title("Volumetric soil moisture")
        fig.autofmt_xdate()
        plt.show()

    def batchAssimilate(self, observations):
        for obs in observations:
            self.assimilate(obs)
        self.display()
    
    def completeSim(self):
        for member in self.ensemble:
            member.run_till_terminate()

    def moveForward(self, days=1):
        for member in self.ensemble:
            member.run(days)
