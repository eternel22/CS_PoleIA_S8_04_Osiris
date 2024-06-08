import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import csv
import pandas as pd

class Spatial_grid:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def interpolate(self, method = 'linear', n = 100):
        xi = np.linspace(min(self.x), max(self.x), n)
        yi = np.linspace(min(self.y), max(self.y), n)
        X, Y = np.meshgrid(xi, yi)
        Z = griddata((self.x, self.y), self.z, (X, Y), method=method)
        return X, Y, Z

    def plot(self, method = 'linear', fig_size = (10, 10), cmap = 'viridis', color_bar = False, elev = 30, azim = 45, box_aspect = [1, 1, 1.1]):
        X, Y, Z = self.interpolate(method)
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111, projection='3d')
        if not color_bar:
            ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cmap, edgecolor='none')
        else:
            surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cmap, edgecolor='none')
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        ax.scatter(self.x, self.y, self.z, color='r', s=50, zorder=10)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('Soil Moisture')
        ax.set_title('Soil Moisture {} interpolation'.format(method))
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect(box_aspect)
        plt.show()

    def cell_size(self):
        X, Y, Z = self.interpolate()
        dx = np.mean(np.diff(X[0, :]))
        dy = np.mean(np.diff(Y[:, 0]))
        return dx, dy
    
    def get_values(self, lin_neigh = True):
        if lin_neigh:
            X, Y, Z = self.interpolate_linear_neighbor()
        else:
            X, Y, Z = self.interpolate()
        return X[0,:], Y[:,0], Z
    
    def interpolate_linear_neighbor(self, n = 100):
        xi = np.linspace(min(self.x), max(self.x), n)
        yi = np.linspace(min(self.y), max(self.y), n)
        X, Y = np.meshgrid(xi, yi)
        Z_lin = griddata((self.x, self.y), self.z, (X, Y), method='linear')
        Z_neigh = griddata((self.x, self.y), self.z, (X, Y), method='nearest')
        for i in range(len(X[0,:])):
            for j in range(len(Y[:,0])):
                if np.isnan(Z_lin[i,j]):
                    Z_lin[i,j] = Z_neigh[i,j]
        return X, Y, Z_lin

    def plot_lin_neigh(self, fig_size = (10, 10), cmap = 'viridis', color_bar = False, elev = 30, azim = 45, box_aspect = [1, 1, 1.1]):
        X, Y, Z = self.interpolate_linear_neighbor()
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111, projection='3d')
        if not color_bar:
            ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cmap, edgecolor='none')
        else:
            surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cmap, edgecolor='none')
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        ax.scatter(self.x, self.y, self.z, color='r', s=50, zorder=10)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('Soil Moisture')
        ax.set_title('Soil Moisture combined interpolation')
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect(box_aspect)
        plt.show()

class HumidityData:
    def __init__(self, file_path):
        self.file_path = file_path
        self.time = []
        for k in range(1, 7):
            setattr(self, f'humidity{k}', [])

    def read_data(self):
        data = pd.read_csv(self.file_path, sep=';')
        self.time = data['Date/heure'].tolist()
        for k in range(1, 7):
            setattr(self, f'humidity{k}', data[f'EAG Humidité du sol {k} [%]'].tolist())

    def get_humidity_values(self):
        return {'time': self.time, 'humidity1': self.humidity1, 'humidity2': self.humidity2, 'humidity3': self.humidity3,
                'humidity4': self.humidity4, 'humidity5': self.humidity5, 'humidity6': self.humidity6}

