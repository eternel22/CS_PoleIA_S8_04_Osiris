import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import datetime as dt
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

# Obervations
dates_of_observation = [dt.date(2000, 2, 7), dt.date(2000, 2, 28), dt.date(2000, 3, 20),
                        dt.date(2000, 4, 10), dt.date(2000, 5, 1)]
observed_lai = np.array([2.2, 3.5, 6.2, 3.3, 2.1])
observed_sm = np.array([0.285, 0.26, 0.28, 0.18, 0.17])
std_lai = observed_lai * 0.1  # Std. devation is estimated as 10% of observed value
std_sm = observed_sm * 0.05  # Std. devation is estimated as 5% of observed value
observed_state = np.stack([observed_lai, observed_sm])
cov_matrix = np.zeros((len(dates_of_observation), 2, 2))
cov_matrix[:, 0, 0] = std_lai ** 2
cov_matrix[:, 1, 1] = std_sm ** 2

# Initial estimation
estimated_lai = np.zeros((len(dates_of_observation),))
estimated_sm = np.zeros((len(dates_of_observation),))
model = Wofost72_WLP_FD(parameters, weather, agromanagement)
for i, d in enumerate(dates_of_observation):
    model.run_till(d)
    estimated_lai[i] = model.get_variable("LAI")
    estimated_sm[i] = model.get_variable("SM")
    # print(f"Estimated LAI at {d}: {estimated_lai[i]}")
    # print(f"Estimated SM at {d}: {estimated_sm[i]}")
estimated_state = np.stack([estimated_lai, estimated_sm])

# Pack them into a convenient format
observations_for_DA = [(d, {"LAI": (lai, errlai), "SM": (sm, errsm)}) for d, lai, errlai, sm, errsm in zip(
    dates_of_observation, observed_lai, std_lai, observed_sm, std_sm)]


# # compute the cost function of 4DVAR
def cost_function(estimated_state, observed_state, cov_matrix):
    J = 0
    for i in range(estimated_state.shape[1]):
        J += 1/2 * (observed_state[:, i]-estimated_state[:, i]).T @ \
            np.linalg.inv(
                cov_matrix[i]) @ (observed_state[:, i]-estimated_state[:, i])
    return J


J = cost_function(estimated_state, observed_state, cov_matrix)
print(f"Initial cost function: {J:.3f}")
print(observed_state[:, 0].shape)


corrected_initial_state = {"LAI": 0.12, "SM": 0.15}
corrected_lai = np.zeros((len(dates_of_observation),))
corrected_sm = np.zeros((len(dates_of_observation),))
corrected_model = Wofost72_WLP_FD(parameters, weather, agromanagement)
for k, v in corrected_initial_state.items():
    _ = corrected_model.set_variable(k, v)

for i, d in enumerate(dates_of_observation):
    corrected_model.run_till(d)
    corrected_lai[i] = corrected_model.get_variable("LAI")
    corrected_sm[i] = corrected_model.get_variable("SM")
    # print(f"Estimated LAI at {d}: {estimated_lai[i]}")
    # print(f"Estimated SM at {d}: {estimated_sm[i]}")
corrected_state = np.stack([corrected_lai, corrected_sm])
corrected_J = cost_function(corrected_state, observed_state, cov_matrix)
print(f"Corrected cost function: {corrected_J:.3f}")


# plot
plt.figure(1)
plt.plot(dates_of_observation, estimated_lai, label="Estimated LAI")
plt.scatter(dates_of_observation, observed_lai, label="Observed LAI")
plt.plot(dates_of_observation, corrected_lai, label="Corrected LAI")
# limit to 3 digits

plt.title("Initial state "+f"J={J:.3f}" +
          " Corrected state "+f"J={corrected_J:.3f}")

plt.figure(2)
plt.plot(dates_of_observation, estimated_sm, label="Estimated SM")
plt.scatter(dates_of_observation, observed_sm, label="Observed SM")
plt.plot(dates_of_observation, corrected_sm, label="Corrected SM")
plt.title("Initial state "+f"J={J:.3f}" +
          " Corrected state "+f"J={corrected_J:.3f}")

plt.legend()
plt.show()
