import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
import pandas as pd
from pcse.models import Wofost72_WLP_FD
import kalmanWoFost
import copy
from sklearn.metrics import mean_squared_error



class LSTM_WOFOST:
    def __init__(self,ensemble_size, parameters, weather, agromanagement, override_parameters):
        self.__parameters = parameters
        self.__agromanagement = agromanagement
        self.__weather = weather
        self.__ensemble_size = ensemble_size
        self._observations = {}
        self.__override_parameters = override_parameters
        self.parameter_set = []
        self.w_set = {}
        self.kf_set = {}

        # Init the LSTM
        self.model = Sequential()
        self.length = 70
        self.model.add(LSTM(10, input_shape=(1,4)))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.training_size = len(override_parameters[list(override_parameters.keys())[0]])
        
        self.x_set = []
        self.y_set = []

    def initializeSets(self):
        for iT in range(self.training_size):
            
            p = copy.deepcopy(self.__parameters)
            for par, distr in self.__override_parameters.items():
                # This is were the parameters are modified using the set_override command
                p.set_override(par, distr[iT])
            member = Wofost72_WLP_FD(p, self.__weather, self.__agromanagement)
            try:
                member.run_till_terminate()
                print("====",iT)
                temp = pd.DataFrame(member.get_output())
                self.y_set.append(list(temp['SM'])[-70:])
                self.x_set.append([distr[iT] for par, distr in self.__override_parameters.items()])
            except:
                print("[LSTM] Failed to compute for iteration {}".format([distr[iT] for par, distr in self.__override_parameters.items()]))
        self.x_set = np.array(self.x_set)
        self.x_set = np.reshape(self.x_set, (self.x_set.shape[0], 1, self.x_set.shape[1]))
        self.y_set = np.array(self.y_set)
        print('x size: {}'.format(self.x_set.shape))
        print('y size: {}'.format(self.y_set.shape))
    
    def train(self):
        self.model.fit(self.x_set, self.y_set, epochs=100, batch_size=1, verbose=2)
    
    def test(self):
        print(self.model.predict(self.x_set).shape)
        return np.sqrt(mean_squared_error(self.y_set[0], self.model.predict(self.x_set)))



# # Reshape the data for LSTM input
# x_train = x_train.reshape((x_train.shape[0], 1, 1))

# # Create the LSTM model
# model = Sequential()
# model.add(LSTM(10, input_shape=(1, 1)))
# model.add(Dense(1))

# # Compile and train the model
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(x_train, y_train, epochs=10, batch_size=32)

# # Generate test data
# x_test = np.linspace(0, 10, 100)
# y_test = model_function(x_test)

# # Reshape the data for LSTM input
# x_test = x_test.reshape((x_test.shape[0], 1, 1))

# # Predict the output using the trained model
# y_pred = model.predict(x_test)

# # Print the predicted and actual values
# for i in range(len(x_test)):
#     print(f'Input: {x_test[i][0][0]}, Predicted: {y_pred[i][0]}, Actual: {y_test[i]}')