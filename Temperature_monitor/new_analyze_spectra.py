import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from wanglab_instruments.wanglab_instruments.utils import helpers as h
import os
import datetime

#spectrum analyzer data
folder = 'spectrum_analyzer_data20200915_HOAtest0_51uHthickcoil/'
_format = '%Y%m%d_%H%M%S.npy'

#temperature data
filename = 'MI_TempDataLog_09_16_2020.csv'
startdate = '20200306'
temp_data = np.loadtxt(filename, delimiter=",", skiprows = 6, dtype={'names': ('time', 'temp'),'formats': ('|U16', np.float)})
temp_temps = np.loadtxt(filename, delimiter=",", skiprows=6, usecols=1)

plot_every = 10
avg_int = 1*60 #avg data within 2 min intervals
end_index = 2300
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


colors = cm.rainbow(np.linspace(0, 1, len(times_sorted)/plot_every + 5))
plot = 0
for i in range(len(times_sorted)):
    dt.append((times_sorted[i] - times_sorted[0]).total_seconds())

    if i%plot_every == 0: #plot every nth curve
        plot += 1
        plt.plot(data_sorted[i][0],data_sorted[i][1], color = colors[-plot])
plt.xlabel("Frequency")
plt.ylabel("Reflection power dBm") #check these
plt.show()


plt.subplot(3,1,1)
plt.plot(dt,mins_sorted, '.')
plt.ylabel('Reflection dip (dBm)')

plt.subplot(3,1,2)
plt.plot(dt, Qs_sorted, ".")
plt.ylabel('Quality factor')

plt.subplot(3,1,3)
plt.plot(dt, freqs_sorted, '.')
plt.xlabel('Time (s)')
plt.ylabel('Center frequency (MHz)')

plt.show()


#temp analysis
add_date = datetime.datetime.strptime(startdate, "%Y%m%d")
day = datetime.timedelta(days=1)

temp_time = []
temp_datetime = []
temp_dt = []

for i in range(0,len(temp_data)):
    temp_time.append(datetime.datetime.strptime(temp_data[i][0], "%H:%M:%S"))
    if i==0:
        temp_datetime.append(add_date + datetime.timedelta(hours=temp_time[i].hour,
                          minutes=temp_time[i].minute, seconds=temp_time[i].second))
    if i>0:
        if temp_time[i] - temp_time[i-1] < datetime.timedelta(seconds=0):
            temp_datetime.append(add_date + datetime.timedelta(hours=temp_time[i].hour,
                          minutes=temp_time[i].minute, seconds=temp_time[i].second) + day)
        else:
            temp_datetime.append(add_date + datetime.timedelta(hours=temp_time[i].hour,
                          minutes=temp_time[i].minute, seconds=temp_time[i].second))
    temp_dt.append((temp_datetime[i] - temp_datetime[0]).total_seconds())


#plot every avg_int time period
dt = np.array(dt)
temp_dt = np.array(temp_dt)
mins_sorted = np.array(mins_sorted)
Qs_sorted = np.array(Qs_sorted)
freqs_sorted = np.array(freqs_sorted)
print(len(temp_dt))
temp_dt = temp_dt[0:end_index]

time_avg = np.zeros(int(temp_dt[-1]/avg_int))
temp_avg = np.zeros(int(temp_dt[-1]/avg_int))
min_avg = np.zeros(int(temp_dt[-1]/avg_int))
freq_avg = np.zeros(int(temp_dt[-1]/avg_int))
Qs_avg = np.zeros(int(temp_dt[-1]/avg_int))

for i in range(0,int(temp_dt[-1]/avg_int)):
    time_avg[i] = (avg_int*(i+1) + avg_int*(i))/2
    temp_avg[i] = np.average(temp_temps[np.where(np.logical_and(temp_dt > avg_int*i, temp_dt < avg_int*(i+1)))[0]])
    min_avg[i] = np.average(mins_sorted[np.where(np.logical_and(dt > avg_int*i, dt < avg_int*(i+1)))][0])
    Qs_avg[i] = np.average(Qs_sorted[np.where(np.logical_and(dt > avg_int*i, dt < avg_int*(i+1)))][0])
    freq_avg[i] = np.average(freqs_sorted[np.where(np.logical_and(dt > avg_int*i, dt < avg_int*(i+1)))][0])

plt.subplot(3,1,1)
plt.plot(temp_avg,min_avg, '.')
plt.ylabel('Reflection dip (dBm)')

plt.subplot(3,1,2)
plt.plot(temp_avg, Qs_avg, ".")
plt.ylabel('Quality factor')

plt.subplot(3,1,3)
plt.plot(temp_avg, freq_avg, '.')
plt.xlabel('Temp (K)')
plt.ylabel('Center frequency (MHz)')

plt.show()

