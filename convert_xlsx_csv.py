import pandas as pd
import pandas as pd
import os
"""
path = 'DATA/Probes/'
path1 = path + 'Sonde Canon 2.xlsx'

df = pd.read_excel(path1)
# Delete the first line of df

# Convert to CSV file
csv_path = 'DATA/Probes/Sonde Canon 2.csv'
df.to_csv(csv_path, index=False, sep=';')

df_csv = pd.read_csv(csv_path, skiprows=1)

first_line = pd.read_csv(csv_path).iloc[0]

df_csv.to_csv(csv_path, index=False)
df_csv = pd.read_csv(csv_path)

if 'Température de l\'air [°C]' in first_line:
    df_csv = df_csv.iloc[:, [0] + list(range(4, len(df_csv.columns)))]
if 'Température sèche [°C]' in first_line:
    df_csv = df_csv.iloc[:, [0] + list(range(4, len(df_csv.columns)))]

header = ['Date/heure','Précipitations [mm]','EAG Humidité du sol 1 [%]',
          'EAG Humidité du sol 2 [%]','EAG Humidité du sol 3 [%]','EAG Humidité du sol 4 [%]','EAG Humidité du sol 5 [%]',
          'EAG Humidité du sol 6 [%]','Température du sol MOY 1 [°C]','Température du sol MAX 1 [°C]','Température du sol MIN 1 [°C]',
          'Température du sol MOY 2 [°C]','Température du sol MAX 2 [°C]','Température du sol MIN 2 [°C]','Température du sol MOY 3 [°C]',
          'Température du sol MAX 3 [°C]','Température du sol MIN 3 [°C]','Température du sol MOY 4 [°C]','Température du sol MAX 4 [°C]',
          'Température du sol MIN 4 [°C]','Température du sol MOY 5 [°C]','Température du sol MAX 5 [°C]','Température du sol MIN 5 [°C]',
          'Température du sol MOY 6 [°C]','Température du sol MAX 6 [°C]','Température du sol MIN 6 [°C]',
          'Panneau solaire [mV]','Batterie [mV]']

header_to_str = ';'.join(header)
df_csv.columns = [header_to_str]

# Save the modified CSV file
df_csv.to_csv(csv_path, index=False)
"""
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
        
        first_line = pd.read_csv(csv_path).iloc[0]
        df_csv = pd.read_csv(csv_path, skiprows=1)

        df_csv.to_csv(csv_path, index=False)
        df_csv = pd.read_csv(csv_path)

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

        header_to_str = ';'.join(header)
        df_csv.columns = [header_to_str]

        df_csv.to_csv(csv_path, index=False)

    def folder_convert(self):
        for file in os.listdir(self.path):
            if file.endswith('.xlsx'):
                self.convert_xlsx_to_csv(file)
            else:
                continue

# Usage example
path = 'DATA/Probes/'

convert = Converter(path)
convert.folder_convert()
