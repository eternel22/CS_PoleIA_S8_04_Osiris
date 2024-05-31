from pcse.models import Wofost72_WLP_FD, Wofost72_PP
import matplotlib.pyplot as plt

def getWofost_WaterLimited(parameters, weatherdataprovider, agromanagement):
    return Wofost72_WLP_FD(parameters, weatherdataprovider, agromanagement)

def getWofost_PotentialProd(parameters, weatherdataprovider, agromanagement):
    return Wofost72_PP(parameters, weatherdataprovider, agromanagement)

def plotWofostDF(df):
    fig, axes = plt.subplots(nrows=11, ncols=1, figsize=(20,40), sharex=True)
    for var, ax in zip(["DVS","LAI","TAGP","TWSO","TWLV","TWST","TWRT","TRA","RD","SM","WWLOW"], axes.flatten()):
        ax.plot_date(df.index, df[var], 'b-')
        ax.set_title(var)
    fig.autofmt_xdate()