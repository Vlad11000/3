import matplotlib.pyplot as plt
import argparse
from scipy.io import netcdf_file
import numpy as np
import json
parser = argparse.ArgumentParser()
parser.add_argument('longitude', metavar='LON', type=float, help='Longitude, deg')
parser.add_argument('latitude',  metavar='LAT', type=float, help='Latitude, deg')
if __name__ == "__main__":
    args = parser.parse_args()
    with netcdf_file("D:\\2 курс 2 семестр\\ОАД\\MSR-2.nc", "r") as dataset:
        LON, LAT = args.longitude, args.latitude
        ind_LON = np.searchsorted(dataset.variables['longitude'].q, LON)
        ind_LAT = np.searchsorted(dataset.variables['latitude'].q, LAT)
        time = dataset.variables['time'].q.copy()
        ozon = dataset.variables['Average_O3_column'].q[:, ind_LAT, ind_LON].copy()
    min_ozon = np.min(ozon)
    max_ozon = np.max(ozon)
    mean_ozon = np.mean(ozon)
    july_ozon = ozon[6::12]
    july_max_ozon = np.max(july_ozon)
    july_min_ozon = np.min(july_ozon)
    july_mean_ozon = np.mean(july_ozon)
    jan_ozon = ozon[::12]
    jan_max_ozon = np.max(jan_ozon)
    jan_min_ozon = np.min(jan_ozon)
    jan_mean_ozon = np.mean(jan_ozon)
q = {
    "coordinates": [LON, LAT],
    "january":{
        "min": float(jan_min_ozon),"max": float(jan_max_ozon),"mean": float(jan_mean_ozon)
     },
    "july":{
        "min": float(july_min_ozon),"mean": float(july_mean_ozon), "max": float(july_max_ozon),
   },
     "all": {
    "min": float(min_ozon),"max": float(max_ozon),"mean": float(mean_ozon)
   }
    }
plt.xlabel('Time')
plt.ylabel('Ozon')
plt.plot(time, ozon, label='ozon(all)')
plt.plot(time[::12], jan_ozon, label='ozon(january)')
plt.plot(time[6::12], july_ozon, label='ozon(july)')
plt.legend()
plt.savefig('ozon.jpg')
with open("ozon.json", "w") as datafile:
        json.dump(q, datafile)