import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.dates as dt
from matplotlib.dates import strpdate2num
from wanglab_instruments.wanglab_instruments.utils import helpers as h
import os
import datetime
import math
from matplotlib import dates


#spectrum analyzer data
folder = 'CoolDownApr12/'
_format = '%Y%m%d_%H%M%S.npy'

#temperature data
filename = 'MI_TempDataLog 04_12_2021.csv'
# startdate = '20200916'
start_row = 62
end_row = 36962
diff = end_row-start_row
interval = 27 
#the spectrum data have different time intervals from temperature data, the 'interval' here is the difference.
# temp_times = np.loadtxt(filename, delimiter=",", converters = {1: lambda s: float(s.replace(":",""))}, skiprows=start_row-1, usecols=1)[:diff] 
temp_temps = np.loadtxt(filename, delimiter=",", skiprows=start_row-1, usecols=3)[:diff] 
print(temp_temps)
skipping_temps = temp_temps[::interval]
print('length of Skipping_temps = ',len(skipping_temps))

plt.subplot(1,1,1)
plt.plot(skipping_temps, '.')
plt.xlabel('Time')
plt.ylabel('Temperature (K)')
plt.show()


plot_every = 2 # plot the reflection curves every other n data
avg_int = 2*60 #avg data within 2 min intervals
end_index = 10000
data_list = []
data_times = []
times = []

for root, dirs, fils in os.walk(folder):
    for fil in fils:
        data_list.append(np.load(os.path.join(root,fil)))
        data_times.append(fil.strip('.npy'))
        times.append(datetime.datetime.strptime(fil,_format))

mins = []
freqs = []
Qs = []

for i in range(len(data_list)):
    xdat = data_list[i][0] #list of all tested frequencies at given time snapshot
    ydat = data_list[i][1] #gain at every frequency at given time snapshot
    baseline = ydat[0]
    mins.append(np.min(ydat))
    freqs.append(xdat[ydat==np.min(ydat)][0]) #find minimum of reflection dip, record freq
    popt, pcov = h.fit_lorentzian(xdat, h.unlog(ydat),x0=freqs[-1],
                        amp =-1)
    Qs.append(np.abs(popt[0]/popt[-1]))

mins = np.array(mins)
Qs = np.array(Qs)
freqs = np.array(freqs)

tsort = np.array(sorted(range(len(times)), key=lambda k: times[k])) #get key to sort all data by times

times_sorted = [times[i] for i in tsort] #sort all data by times, these times are still datetime objects
mins_sorted = [mins[i] for i in tsort]
freqs_sorted = [freqs[i] for i in tsort]
Qs_sorted = [Qs[i] for i in tsort]
data_sorted = [data_list[i][:] for i in tsort]

dt = []  #find the total time elapsed from start, in seconds (type: float)


colors = cm.rainbow(np.linspace(0, 1, math.ceil(len(times_sorted)/plot_every + 5)))
plot = 0
for i in range(len(times_sorted)-5):
    dt.append((times_sorted[i] - times_sorted[0]).total_seconds())

    if i%plot_every == 0: #plot every nth curve
        plot += 1
        plt.plot(data_sorted[i][0],data_sorted[i][1], color = colors[-plot])
plt.xlabel("Frequency")
plt.ylabel("Reflection power (dBm)") #check these
plt.show()

nn = len(dt)-5

dtmin = [x / 60 for x in dt]
plt.subplot(3,1,1)
plt.plot(dtmin[:nn],mins_sorted[:nn], '.')
plt.ylabel('Reflection dip (dBm)')

plt.subplot(3,1,2)
plt.plot(dtmin[:nn], Qs_sorted[:nn], ".")
plt.ylabel('Quality factor')

plt.subplot(3,1,3)
plt.plot(dtmin[:nn], freqs_sorted[:nn], '.')
plt.xlabel('Time (s)')
plt.ylabel('Center frequency (MHz)')

plt.show()


#temp analysis
plt.subplot(3,1,1)
plt.plot(skipping_temps[:nn],mins_sorted[:nn], '.')
plt.ylabel('Reflection dip (dBm)')

plt.subplot(3,1,2)
plt.plot(skipping_temps[:nn], Qs_sorted[:nn], ".")
plt.ylabel('Quality factor')

plt.subplot(3,1,3)
plt.plot(skipping_temps[:nn], freqs_sorted[:nn], '.')
plt.xlabel('Temperature (K)')
plt.ylabel('Center frequency (MHz)')

plt.show()
