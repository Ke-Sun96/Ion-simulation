import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import dates
from wanglab_instruments.wanglab_instruments.utils import helpers as h
import os
import datetime
import math

folder = 'CoolDownApr12/'
_format = '%Y%m%d_%H%M%S.npy'
_format2 = '%Y%m%d_%H%M%S'
data_list = []
data_times = []
times = []


for root, dirs, fils in os.walk(folder):
    for fil in fils:
        data_list.append(np.load(os.path.join(root,fil)))
        data_times.append(fil.strip('.npy'))
        times.append(datetime.datetime.strptime(fil,_format))

dtimes = [datetime.datetime.strptime(d, _format2) for d in data_times]
dnums = (dates.date2num(dtimes) -np.min(dates.date2num(dtimes)))*24*60

mins = []
maxs = []
freqs = []
Qs = []

#analyzeNum = 10
ErrorCount = 0
for i in range(len(data_list)):
#for i in range(analyzeNum):
    xdat = data_list[i][0]
    ydat = data_list[i][1]
    baseline = ydat[0]
    mins.append(np.min(ydat))
    maxs.append(np.max(ydat))
    freqs.append(xdat[ydat==np.min(ydat)][0])
    try:	
    	popt, pcov = h.fit_lorentzian(xdat, h.unlog(ydat),x0=freqs[-1], 
                        amp = -1)
    except RuntimeError:
    	ErrorCount +=1
    	print('Cannot fit #',ErrorCount)
    Qs.append(popt[0]/np.abs(popt[-1]))
    # print(freqs,Qs,mins)
''' 
    if i%20 == 0:
        l = math.floor(len(xdat)/1)
        plt.plot(xdat[0:l],ydat[0:l])
        plt.ylabel('Power(dBm)')
        plt.xlabel('Frequency (MHz)')
       # plt.plot(xdat,10*np.log10(h.lorentzian(xdat,*popt)))
plt.show()
'''
mins = np.array(mins)
maxs = np.array(maxs)
Qs = np.array(Qs)
freqs = np.array(freqs)
plot_every = 1

# sort the time
tsort = np.array(sorted(range(len(times)), key=lambda k:times[k]))
time_sorted = [times[i] for i in tsort]
t_sorted = (dates.date2num(time_sorted) -
np.min(dates.date2num(time_sorted)))*24*60
mins_sorted = [mins[i] for i in tsort]
freqs_sorted = [freqs[i] for i in tsort]
Qs_sorted = [Qs[i] for i in tsort]
maxs_sorted = [maxs[i] for i in tsort]
data_sorted = [data_list[i][:] for i in tsort]

plot_before = len(time_sorted) # 2*mins
# print(len(time_sorted))
# add time filter
filt = dnums<plot_before

dt = [] # find the total time elapsed from start, in seconds (type: float)
colors = cm.rainbow(np.linspace(0,1,math.ceil(plot_before/plot_every+5)))
plot = 0
for i in range(len(time_sorted)):
    dt.append((time_sorted[i]-time_sorted[i]).total_seconds())

    if i%plot_every == 0 and i < plot_before:
        plot+=1
        plt.plot(data_sorted[i][0], data_sorted[i][1],color = colors[plot])
plt.xlabel('Frequency (MHz)')
plt.ylabel('Power (dBm)')
plt.show()

plt.subplot(3,1,1)
plt.plot(dnums[filt],mins[filt],'.')
plt.ylabel('Reflection dip (dBm)')
plt.subplot(3,1,2)
plt.plot(dnums[filt],Qs[filt],'.')
plt.ylabel('Quality factor')
plt.subplot(3,1,3)
plt.plot(dnums[filt],freqs[filt],'.')
plt.xlabel('Time (min)') 
plt.ylabel('Center frequency (MHz)')
plt.show()
