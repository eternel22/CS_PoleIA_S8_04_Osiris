import pandas as pd
import os

class Converter:
    def __init__(self, path):
        self.path = path

    def convert_xlsx_to_csv(self, file = None):
        path = self.path
        if file is not None:
            path = path + file
            
        df = pd.read_excel(path)
        df = df.iloc[1:]  # Delete the first line of df
        
        csv_path = path.replace('.xlsx', '.csv')
        df.to_csv(csv_path, index=False, sep=';')
        
        first_line = pd.read_csv(csv_path,sep=';').columns
        
        df_csv = pd.read_csv(csv_path, skiprows=0, sep=';')

        df_csv.to_csv(csv_path, index=False, sep = ';')
        df_csv = pd.read_csv(csv_path, sep=';')
        if 'Température de l\'air [°C]' in first_line:
            df_csv = df_csv.iloc[:, [0] + list(range(4, len(df_csv.columns)))]
        if 'Température sèche [°C]' in first_line:
            df_csv = df_csv.iloc[:, [0] + list(range(4, len(df_csv.columns)))]

        header = ['Date/heure', 'Précipitations [mm]', 'EAG Humidité du sol 1 [%]',
                  'EAG Humidité du sol 2 [%]', 'EAG Humidité du sol 3 [%]', 'EAG Humidité du sol 4 [%]',
                  'EAG Humidité du sol 5 [%]', 'EAG Humidité du sol 6 [%]', 'Température du sol MOY 1 [°C]',
                  'Température du sol MAX 1 [°C]', 'Température du sol MIN 1 [°C]', 'Température du sol MOY 2 [°C]',
                  'Température du sol MAX 2 [°C]', 'Température du sol MIN 2 [°C]', 'Température du sol MOY 3 [°C]',
                  'Température du sol MAX 3 [°C]', 'Température du sol MIN 3 [°C]', 'Température du sol MOY 4 [°C]',
                  'Température du sol MAX 4 [°C]', 'Température du sol MIN 4 [°C]', 'Température du sol MOY 5 [°C]',
                  'Température du sol MAX 5 [°C]', 'Température du sol MIN 5 [°C]', 'Température du sol MOY 6 [°C]',
                  'Température du sol MAX 6 [°C]', 'Température du sol MIN 6 [°C]', 'Panneau solaire [mV]',
                  'Batterie [mV]']

        df_csv.columns = header
        df_csv = df_csv.sort_values(by='Date/heure').reset_index(drop=True)
        df_csv.to_csv(csv_path, index=False, sep=';')

    def folder_convert(self):
        for file in os.listdir(self.path):
            if file.endswith('.xlsx'):
                self.convert_xlsx_to_csv(file)
            else:
                continue

# Usage example
path = 'data/Data 2022/Probes'

convert = Converter(path)
convert.folder_convert()
