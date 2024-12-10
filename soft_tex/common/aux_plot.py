import matplotlib.pyplot as plt

FONT_SIZE   = 16
TICK_SIZE   = 16
LEGEND_SIZE = 16
AXES_SIZE   = 16

plt.rc('font',   size=FONT_SIZE)           # controls default text sizes
plt.rc('axes',   labelsize=AXES_SIZE)      # font size of the x and y labels
plt.rc('xtick',  labelsize=TICK_SIZE)      # font size of the tick labels
plt.rc('ytick',  labelsize=TICK_SIZE)      # font size of the tick labels
plt.rc('legend', fontsize=LEGEND_SIZE)     # legend font size