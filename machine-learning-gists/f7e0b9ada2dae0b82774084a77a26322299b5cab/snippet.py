## https://github.com/EpistasisLab/penn-ml-benchmarks

## pip install pmlb

import numpy as np
from pmlb import fetch_data
from pmlb import dataset_names

x = np.zeros(len(dataset_names))
for i, dn in enumerate(dataset_names):
    d = fetch_data(dn)
    n = d.describe()["class"]["count"] 
    x[i] = n
    print(str(n) + "   " + str(dn))

x.min()
np.percentile(x, 50)
np.percentile(x, 80)
np.percentile(x, 90)
x.max()


#In [6]: x.min()
#Out[6]: 32.0
#
#In [7]: np.percentile(x, 50)
#Out[7]: 690.0
#
#In [8]: np.percentile(x, 80)
#Out[8]: 3772.0
#
#In [9]: np.percentile(x, 90)
#Out[9]: 7400.0
#
#In [10]: x.max()
#Out[10]: 1025009.0




## Largest datasets:

#19020.0   magic
#20000.0   letter
#28056.0   krkopt
#48842.0   adult
#58000.0   shuttle
#67557.0   connect-4
#70000.0   mnist
#100968.0   fars
#105908.0   sleep
#494020.0   kddcup
#1025009.0   poker




