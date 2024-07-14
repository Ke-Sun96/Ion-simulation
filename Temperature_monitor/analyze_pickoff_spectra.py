import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import dates
from wanglab_instruments.wanglab_instruments.utils import helpers as h
import os
import datetime

folder = 'spectrum_analyzer_data20200316TestSmallToroid470pF_WithAmp_PCBTemp/'
_format = '%Y%m%d_%H%M%S.npy'
_format2 = '%Y%m%d_%H%M%S'
data_list = []
data_times = []
times = []

i = 0
for root, dirs, fils in os.walk(folder):
    for fil in fils:
        data_list.append(np.load(os.path.join(root,fil)))
        data_times.append(fil.strip('.npy'))
        times.append(datetime.datetime.strptime(fil,_format))

dtimes = [datetime.datetime.strptime(d, _format2) for d in data_times]
dnums = (dates.date2num(dtimes) - np.min(dates.date2num(dtimes)))*24*60

mins = []
maxs = []
freqs = []
Qs = []

_flag = True
for i in range(len(data_list)):
    xdat = data_list[i][0]
    ydat = data_list[i][1]
    baseline = ydat[0]
    mins.append(np.min(ydat))
    maxs.append(np.max(ydat))
    freqs.append(xdat[ydat==np.max(ydat)][0])
    popt, pcov = h.fit_lorentzian(xdat, h.unlog(ydat))
    Qs.append(popt[0]/np.abs(popt[-1]))
    print(freqs, Qs, maxs)
    '''if popt[0]/popt[-1]:
        plt.plot(xdat, ydat)
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Power (dBm)')
        plt.plot(xdat, 10*np.log10(h.lorentzian(xdat, *popt)))
'''
plt.show()

mins = np.array(mins)
maxs = np.array(maxs)
Qs = np.array(Qs)
freqs = np.array(freqs)
plot_every = 1

# sort the time
tsort = np.array(sorted(range(len(times)),key = lambda k:times[k]))
time_sorted = [times[i] for i in tsort]
mins_sorted = [mins[i] for i in tsort]
freqs_sorted = [freqs[i] for i in tsort]
Qs_sorted = [Qs[i] for i in tsort]
maxs_sorted = [maxs[i] for i in tsort]
data_sorted = [data_list[i][:] for i in tsort]
dt = [] # find the total time, in seconds
# colors = cm.rainbow(np.linspace(0,1,len(time_sorted)/plot_every+5))
plot_before = 1000
colors = cm.rainbow(np.linspace(0,1,plot_before/plot_every+5))
plot = 0

for i in range(len(time_sorted)):
    dt.append((time_sorted[i]-time_sorted[0]).total_seconds())
    if i % plot_every == 0 and i<plot_before:
        plot+=1
        plt.plot(data_sorted[i][0],data_sorted[i][1], color = colors[plot])
plt.show()

t = np.array([times[i].hour*60**2
            +times[i].minute*60
            +times[i].second for i in
    range(len(times))])
t = t - np.min(t)
filt = dnums<plot_before/2
'''
plt.subplot(3,1,1)
plt.plot(dnums,mins[filt],'.')
plt.ylabel('Reflection dip (dBm)')
'''
plt.subplot(3,1,1)
plt.plot(dnums[filt],maxs[filt],'.')
plt.ylabel('Pickoff peak (dBm)')
plt.subplot(3,1,2)
plt.plot(dnums[filt],Qs[filt],'.')
plt.ylabel('Quality factor')
plt.subplot(3,1,3)
plt.plot(dnums[filt],freqs[filt],'.')
plt.xlabel('Time (min)')
plt.ylabel('Center frequency (MHz)')
plt.tight_layout()
plt.show()
