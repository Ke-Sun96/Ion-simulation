import numpy as np
import matplotlib.pyplot as plt
from wanglab_instruments.wanglab_instruments import instruments as wl
from prologix.prologix import Prologix
import serial
import datetime
import time
import os

save_folder = 'spectrum_analyzer_data20200317CoolSmallToroid200pF_WithAmp_PCBTemp/'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

_format = '%Y%m%d_%H%M%S'
def file_name(name_format=_format):
    return datetime.datetime.now().strftime(name_format)

plx = Prologix('/dev/ttyUSB0')
rsa = wl.spectrum_analyzers.AgilentESA(plx.open_instrument(gpib_address=18))

def monitor_resonator(delay_time=10):
    while True:
        x, y = rsa.fetch_spectrum(1)
        f_name = file_name()
        np.save(save_folder + f_name, [x,y])
        print('Saved file: {}'.format(save_folder + f_name), end='\r')
        time.sleep(delay_time)

if __name__ == '__main__':
    monitor_resonator()
