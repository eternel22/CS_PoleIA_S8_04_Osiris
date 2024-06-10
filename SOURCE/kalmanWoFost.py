import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pcse.models import Wofost72_WLP_FD
import numpy as np
import pandas as pd
import copy
import matplotlib
matplotlib.style.use('ggplot')

class KalmanWofostDA():

    def __init__(self, ensemble_size, parameters, weather, agromanagement, override_parameters=None):
        self.__parameters = parameters
        self.__agromanagement = agromanagement
        self.__weather = weather
        self.__ensemble_size = ensemble_size
        self._observations = {}
        self.__override_parameters = override_parameters
        self.trainX = []
        self.trainY = []

        self.__initializeWofostNoDA()
        self.__initializeEnsemble()

    
    def __initializeEnsemble(self):
        """
        Initialize the ensemble of models.
        Inputs:
        - None
        Outputs:
        - None
        """
        
        self.ensemble = []

        for i in range(self.__ensemble_size):
            p = copy.deepcopy(self.__parameters)
            if self.__override_parameters == None:
                Warning("[KalmanWoFoStDA] No override parameters given.")
            else:
                for par, distr in self.__override_parameters.items():
                    # This is were the parameters are modified using the set_override command
                    p.set_override(par, distr[i])
            member = Wofost72_WLP_FD(p, self.__weather, self.__agromanagement)
            self.ensemble.append(member)

    def __initializeWofostNoDA(self):
        self.__wofost_noDA = Wofost72_WLP_FD(self.__parameters, self.__weather, self.__agromanagement)
        self.__wofost_noDA.run_till_terminate()


    def batchAssimilate(self, observations):
        """
        Assimilate observations.
        Inputs:
        - observations: list of observations to assimilate.
        Outputs:
        - None

        Observations should be an array which contains tuples :
        - date
        - {"parameter": (value, error)}
        """
        self.__observations_for_DA = observations
        self.__dates_of_observation = [data[0] for data in observations]
        for obs in observations:
            self.assimilate(obs)
        print("[KalmanWoFostDA] {} observations assimilated".format(len(observations)))
    

    def assimilate(self, obs_data):
        """
        Assimilate a single observation.
        Inputs:
        - obs_data: observation to assimilate.
        Outputs:
        - None

        obs_data is a tuple :
        - date
        - {"parameter": (value, error)}
        """
        date = obs_data[0]
        obs = obs_data[1]

        self._observations[date] = obs
        print("[KalmanWoFoStDA] Assimilating data for {} on day {} ".format(str(obs), date))
        variables_for_DA = obs.keys()
        collected_states = []
        for member in self.ensemble:
            member.run_till(date)

        # ==== BEGIN trainX 
        states = []
        for member in self.ensemble:
            temp = pd.DataFrame(member.get_output())
            temp['day'] = pd.to_datetime(temp['day'], format="%Y-%m-%d")
            temp = temp.set_index("day")
            states.append(temp["SM"].to_numpy())
        
        v_t = []
        for date, row in temp.iterrows():
            if(date in self.__dates_of_observation):
                for data in self.__observations_for_DA:
                    if(data[0] != date):
                        continue
                    v_t.append(data[1]["SM"][0])
            else:
                v_t.append(-1)
        states.append(v_t)

        states = np.array(states)
        self.trainX.append(states)
        # ==== END trainX

            
        for member in self.ensemble:
            t = {}
            for state in variables_for_DA:
                t[state] = member.get_variable(state)
            collected_states.append(t)
        df_A = pd.DataFrame(collected_states)
        A = np.matrix(df_A).T
        P_e = np.matrix(df_A.cov())


        perturbed_obs = []
        for state in variables_for_DA:
            (value, std) = obs[state] # both are empiric values
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


        corrected_states = []
        for i in range(len(self.ensemble)):
            member = self.ensemble[i]
            for state in variables_for_DA:
                member.set_variable(state, df_Aa.iloc[i][state])
            corrected_states.append(df_Aa.iloc[i]["SM"])
        self.trainY.append(corrected_states)


    def completeSim(self):
        """
        Run the ensemble until the end of the simulation.
        Inputs:
        - None
        Outputs:
        - None
        """
        for member in self.ensemble:
            member.run_till_terminate()

    def moveForward(self, days=1):
        """
        Increment the simulation by a given number of days.
        Inputs:
        - days: number of days to increment the simulation.
        Outputs:
        - None
        """
        for member in self.ensemble:
            member.run(days)

    def getResultsAllModel(self, columnName):
        """
        Get the results of a given column for all models in the ensemble.
        Inputs:
        - columnName: name of the column to get.
        Outputs:
        - results: array of results.
        """
        results = []
        for member in self.ensemble:
            temp = pd.DataFrame(member.get_output())
            temp['day'] = pd.to_datetime(temp['day'], format="%Y-%m-%d")
            temp = temp.set_index("day")
            results.append(temp[columnName].to_numpy())
        return np.array(results)

    def getResultsWithDA(self):
        """
        Get the results of the ensemble with DA.
        Inputs:
        - None
        Outputs:
        - results: DataFrame containing the results of the ensemble as the mean of all models.
        """
        results = []
        for member in self.ensemble:
            temp = pd.DataFrame(member.get_output())
            temp['day'] = pd.to_datetime(temp['day'], format="%Y-%m-%d")
            temp = temp.set_index("day")
            results.append(temp)
        concat_df = pd.concat(results)
        mean_df = concat_df.groupby(concat_df.index).mean()
        return mean_df

    def getResultsNoDA(self):
        """
        Get the results of the ensemble without DA.
        Inputs:
        - None
        Outputs:
        - results: DataFrame containing the results of the ensemble without DA.
        """
        df_noDA = pd.DataFrame(self.__wofost_noDA.get_output())
        df_noDA['day'] = pd.to_datetime(df_noDA['day'], format="%Y-%m-%d")
        df_noDA = df_noDA.set_index("day")
        return df_noDA


    def displayLAIsM(self, average=False, fig=None, axes=None):
        """
        Display the LAI and Soil Moisture of the ensemble.
        Inputs:
        - average: boolean to display the average of the ensemble.
        - fig: figure to display the results.
        - axes: axes to display the results.
        Outputs:
        - None, but plots the results.
        """

        print("[KalmanWoFoStDA] Displaying data for {} up to day {} ".format(len(self.ensemble), self.ensemble[0].get_variable("day")))
        results = [pd.DataFrame(member.get_output()).set_index("day") for member in self.ensemble]
        if fig == None:
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16,16), sharex=True)
        if not average:
            for member_df in results:
                member_df["LAI"].plot(style="k:", ax=axes[0])
                member_df["SM"].plot(style="k:", ax=axes[1])
        else:
            average = sum(results)/len(results)
            average["LAI"].plot(style="r-", ax=axes[0])
            average["SM"].plot(style="r-", ax=axes[1])
        if self._observations != None:
            val_lai = [element['LAI'][0] for element in self._observations.values()]
            err_lai = [element['LAI'][1] for element in self._observations.values()]
            val_sm = [element['SM'][0] for element in self._observations.values()]
            err_sm = [element['SM'][1] for element in self._observations.values()]
            axes[0].errorbar(self._observations.keys(),list(val_lai),yerr=err_lai, fmt='o')   
            axes[1].errorbar(self._observations.keys(),list(val_sm),yerr=err_sm, fmt='o')   
        # axes[0].errorbar(self._observations.keys(), self._observations.values, fmt="o")
        # axes[1].errorbar(self._observations.keys(), self._observations.values, yerr=std_sm, fmt="o")
        axes[0].set_title("Leaf area index")
        axes[1].set_title("Volumetric soil moisture")
        fig.autofmt_xdate()
        

    
    def getState(self, specific="all"):
        """ 
        Get the state of the model.
        Inputs:
        - specific: specific state to get.
        Outputs:
        - state: state of the model.
        """
        specifics = {"all": [self.__parameters,self.__weather, self.__agromanagement]}
        return specifics[specific]

    def evaluate(self):
        """
        Evaluate the ensemble.
        Inputs:
        - None
        Outputs:
        - DataFrame containing the evaluation of the ensemble.
        """
        def diff(a,b):
            t = 0
            for el in set(a.keys()).intersection(b.keys()):
                t+= (a[el]-b[el])**2
            return t
        globalv = {}
        for iter in self.ensemble:
            globalv[iter] = diff(pd.DataFrame(iter.get_output()).set_index('day')['LAI'],
                                 pd.DataFrame({element:{'LAI':self._observations[element]['LAI'][0], 'SM':self._observations[element]['SM'][0]} for element in self._observations.keys()}).transpose()['LAI'])
        return pd.DataFrame(globalv)
    
    def get_avg_RMSE(self):
        """
        Get the average RMSE of the ensemble.
        Inputs:
        - None
        Outputs:
        - None, but prints the average RMSE of the ensemble.
        """
        val_sm = [element['SM'][0] for element in self._observations.values()]
        val_lai = [element['LAI'][0] for element in self._observations.values()]
        results = [pd.DataFrame(member.get_output()).set_index("day") for member in self.ensemble]
        RMSE_lai = []
        RMSE_sm = []
        
        for member_df in results:
            observed_days = list(self._observations.keys())
            RMSE_sm.append(np.sqrt(np.mean((member_df.loc[observed_days, "SM"] - val_sm)**2)))
            RMSE_lai.append(np.sqrt(np.mean((member_df.loc[observed_days,"LAI"]-val_lai)**2)))
        
        print('AVG RMSE LAI =', np.mean(RMSE_lai), ', AVG RMSE SM =', np.mean(RMSE_sm))
    

