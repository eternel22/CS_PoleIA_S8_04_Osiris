from pcse.fileinput import YAMLCropDataProvider
from pcse.fileinput import CABOFileReader
import os
from pcse.util import WOFOST72SiteDataProvider
from pcse.fileinput import YAMLAgroManagementReader
from pcse.db import NASAPowerWeatherDataProvider
import pandas as pd
import math, random
import numpy as np

class Dataloader:

    def __init__(self, data_dir) -> None:
        self.data_dir = data_dir
    
    def printCropNames(self):
        
        crop_varieties = self.getCropVarieties()
        
        print("Voici la liste des espèces disponibles :")
        print(list(crop_varieties.keys()))

    def printCropVarietes(self, espece):

        crop_varieties = self.getCropVarieties()

        print("Voici la liste des variétés disponibles pour :", espece)
        print(list(crop_varieties[espece]))

    def getCropVarieties(self):
        """
        Return the names of available crops and varieties per crop.

        :return: a dict of type {'crop_name1': ['variety_name1', 'variety_name1', ...],
                         'crop_name2': [...]}
        """
        cropd = YAMLCropDataProvider()
        return cropd.get_crops_varieties()
    
    def getCropData(self, espece, variete):
        
        cropd = YAMLCropDataProvider()
        cropd.set_active_crop(espece, variete)
        return cropd
    
    def readCropData(self, crop_filename):
        
        crop_dir = os.path.join(self.data_dir, "crop") 
        cropd = CABOFileReader(os.path.join(crop_dir, crop_filename))
        return cropd

    def readSoilData(self, soil_filename):

        soil_dir = os.path.join(self.data_dir, 'soil')
        soild = CABOFileReader(os.path.join(soil_dir, soil_filename))
        return soild

    def getSiteData(self, WAV, IFUNRN = 0, NOTINF = 0., SSMAX = 0., SSI = 0., SMLIM = 0.4):
        """
        - IFUNRN    Indicates whether non-infiltrating fraction of rain is a function of storm size (1)
                    or not (0). Default 0
        - NOTINF    Maximum fraction of rain not-infiltrating into the soil [0-1], default 0.
        - SSMAX     Maximum depth of water that can be stored on the soil surface [cm]
        - SSI       Initial depth of water stored on the surface [cm]
        - WAV       Initial amount of water in total soil profile [cm]
        - SMLIM     Initial maximum moisture content in initial rooting depth zone [0-1], default 0.4
        """
        return WOFOST72SiteDataProvider(WAV=WAV, IFUNRN = IFUNRN, NOTINF = NOTINF, SSMAX = SSMAX, SSI = SSI, SMLIM = SMLIM)
    
    def readAgromanagementData(self, agro_filename):

        agro_dir = os.path.join(self.data_dir, 'agro')
        agrod = YAMLAgroManagementReader(os.path.join(agro_dir, agro_filename))
        return agrod

    def getWeatherData(self, lat, lon):
        return NASAPowerWeatherDataProvider(latitude=lat, longitude=lon)

    def readOsirisData(self, filename):
        
        df_obs = pd.read_csv("data/Data 2022/Probes/" + filename, sep=";")
        df_obs['Date/heure'] = pd.to_datetime(df_obs['Date/heure'], format="%Y-%m-%d %H:%M:%S")
        df_obs = df_obs.sort_values(by='Date/heure').reset_index(drop=True)
        for column in df_obs.columns:
            if column not in ['Date/heure', 'Batterie [mV]', 'Panneau solaire [mV]'] :
                df_obs[column] = df_obs[column].str.replace(',', '.').astype(float)

        df_obs['SM'] = ((df_obs['EAG Humidité du sol 2 [%]'] + df_obs['EAG Humidité du sol 2 [%]'])/2) / 100

        return df_obs
    
    def getOsirisSM(self, filename = 'Sonde Rampe 1.csv', timedelta = pd.Timedelta(1, "d"), error = 0.0, begin = pd.Timestamp("2022-06-04"), end = pd.Timestamp("2022-09-02")):
        """
        Retourne deux tableaux
        - date_of_observation
        - observed_sm
        
        """
        df_obs = self.readOsirisData(filename)

        dates_of_observation = []
        observed_sm = []

        for index, row in df_obs.iterrows():
            current_date = row['Date/heure']
            sm = row['SM']

            if current_date < begin or current_date > end:
                continue
            if len(dates_of_observation) > 0 and current_date - dates_of_observation[-1] < timedelta:
                continue
            if math.isnan(sm):
                continue

            dates_of_observation.append(current_date)
            observed_sm.append(random.uniform(sm * (1-error), sm * (1+error)))


        return dates_of_observation, observed_sm