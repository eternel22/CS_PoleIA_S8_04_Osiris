import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pcse.models import Wofost72_WLP_FD
import numpy as np
import pandas as pd
import copy
import matplotlib
matplotlib.style.use('ggplot')
import kalmanWoFost

class EnKF_LSTM():

    def __init__(self, ensemble_size, parameters, weather, agromanagement, override_parameters=None):
        self.__parameters = parameters
        self.__agromanagement = agromanagement
        self.__weather = weather
        self.__ensemble_size = ensemble_size
        self._observations = {}
        self.__override_parameters = override_parameters



        self.__initializeWofostNoDA()
        self.__initializeEnsemble()

    
    def __initializeEnsemble(self):
        
        self.ensemble = []

        for i in range(self.__ensemble_size):
            p = copy.deepcopy(self.__parameters)
            if self.__override_parameters == None:
                Warning("[EnKF-LSTM] No override parameters given.")
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
        Observations should be an array which contains tuples :
        - date
        - {"parameter": (value, error)}
        """
        self.__observations
        for obs in observations:
            self.assimilate(obs)
        print("[EnKF-LSTM] {} observations assimilated".format(len(observations)))
    

    def assimilate(self, obs_data):
        date = obs_data[0]
        obs = obs_data[1]

        self._observations[date] = obs
        print("[EnKF-LSTM] Assimilating data for {} on day {} ".format(str(obs), date))
        variables_for_DA = obs.keys()
        collected_states = []
        for member in self.ensemble:
            member.run_till(date)
        
        results = []
        for member in self.ensemble:
            temp = pd.DataFrame(member.get_output())
            temp['day'] = pd.to_datetime(temp['day'], format="%Y-%m-%d")
            temp = temp.set_index("day")
            results.append(temp["SM"].to_numpy())
        results = np.array(results)
        
        SMs = np.transpose(results) # (nb_donnees, ensemble_size)
        n_time_actuel = SMs.shape[0]-1

        print("results:", results.shape)
        print("n_time_actuel", n_time_actuel)

        x_lstm = np.array([np.concatenate((SMs, v_t_reshape[:n_time_actuel+1]), axis=1)])

        #print("SMs", SMs.shape)
        print("x_lstm", x_lstm.shape)
        print(x_lstm[0][1])

        y_lstm = models[n_time_actuel].predict(x_lstm)[0]

        print("y_lstm", y_lstm.shape)
        print(y_lstm)
        break
        
        for i in range(len(ensemble)):
            member = ensemble[i]
            print("Member", i," SM", member.get_variable("SM"), ">", y_lstm[i])
            member.set_variable("SM", y_lstm[i])






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

        for i in range(len(self.ensemble)):
            member = self.ensemble[i]
            for state in variables_for_DA:
                member.set_variable(state, df_Aa.iloc[i][state])


    def completeSim(self):
        for member in self.ensemble:
            member.run_till_terminate()

    def moveForward(self, days=1):
        for member in self.ensemble:
            member.run(days)

    def getResultsAllModel(self, columnName):
        results = []
        for member in self.ensemble:
            temp = pd.DataFrame(member.get_output())
            temp['day'] = pd.to_datetime(temp['day'], format="%Y-%m-%d")
            temp = temp.set_index("day")
            results.append(temp[columnName].to_numpy())
        return np.array(results)

    def getResultsWithDA(self):
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
        df_noDA = pd.DataFrame(self.__wofost_noDA.get_output())
        df_noDA['day'] = pd.to_datetime(df_noDA['day'], format="%Y-%m-%d")
        df_noDA = df_noDA.set_index("day")
        return df_noDA


    def displayLAIsM(self, average=False, fig=None, axes=None):
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
        returned_states= []
        specifics = {"all": [self.__parameters,self.__weather, self.__agromanagement]}
        return specifics[specific]

    def evaluate(self):
        def diff(a,b):
            t = 0
            for el in set(a.keys()).intersection(b.keys()):
                t+= (a[el]-b[el])**2
            return t
        globalv = {}
        for iter in k.ensemble:
            globalv[iter] = diff(pd.DataFrame(iter.get_output()).set_index('day')['LAI'],
                                 pd.DataFrame({element:{'LAI':k._observations[element]['LAI'][0], 'SM':k._observations[element]['SM'][0]} for element in k._observations.keys()}).transpose()['LAI'])
        return pd.DataFrame(globalv)
    
    def get_avg_RMSE(self):
        val_sm = [element['SM'][0] for element in self._observations.values()]
        val_lai = [element['LAI'][0] for element in self._observations.values()]
        results = [pd.DataFrame(member.get_output()).set_index("day") for member in self.ensemble]
        RMSE_lai = []
        RMSE_sm = []
        
        for member_df in results:
            observed_days = list(self._observations.keys())
            RMSE_sm.append(np.sqrt(np.mean((member_df.loc[observed_days, "SM"] - val_sm)**2)))
            RMSE_lai.append(np.sqrt(np.mean((member_df.loc[observed_days,"LAI"]-val_lai)**2)))
        
        return print('AVG RMSE LAI =', np.mean(RMSE_lai), ', AVG RMSE SM =', np.mean(RMSE_sm))
    
    def get_RMSE_avg(self):
        val_sm = [element['SM'][0] for element in self._observations.values()]
        val_lai = [element['LAI'][0] for element in self._observations.values()]
        results = [pd.DataFrame(member.get_output()).set_index("day") for member in self.ensemble]
        observed_days = list(self._observations.keys())
        temp_lai= []
        temp_sm = []
        for member_df in results:
            temp_sm.append(member_df.loc[[x + dt.timedelta(days=1) for x in observed_days], "SM"])
            temp_lai.append(member_df.loc[[x + dt.timedelta(days=1) for x in observed_days], "LAI"])
        RMSE_lai = np.sqrt(np.mean((np.mean(temp_lai, axis=0) - val_lai)**2))
        RMSE_sm = np.sqrt(np.mean((np.mean(temp_sm, axis=0) - val_sm)**2))
        return print('RMSE of AVG LAI =', RMSE_lai, ', RMSE of AVG SM =', RMSE_sm)

