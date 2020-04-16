import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import math

# disable divide by zero warnings
import warnings
warnings.filterwarnings("ignore")

SMALL_SIZE = 14
MEDIUM_SIZE = 20
BIGGER_SIZE = 24

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

lwidth = 3

n = 1000

def plot_sub(f,ax1,ax2,x,y,I,title,label_axes=False,plot_dot=False):
    ax1.plot(x,y,lw=lwidth)
    ax1.set_yticks(np.linspace(0, 1, 3))
    ax1.set_title(title)
    if not plot_dot:
        ax2.plot(x,I,lw=lwidth)
    else:
        ax2.plot(x,I,'.',markersize=4*lwidth)
    ax2.set_title(title)
    ax2.set_ylim([0,25])
    ax2.set_xlim((-3,3))
    ax2.set_title('Self-information')
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    if label_axes:
        ax1.set_ylabel('p',rotation=0,labelpad=20)
        ax2.set_ylabel('I(p)',rotation=0,labelpad=20)

### Plot information content function
f = plt.figure(figsize=(7,5))
x = np.linspace(0, 1, n*1e3)
y = -1*np.log2(x)
plt.plot(x,y,lw=lwidth)
plt.title('Self-information I(p)')
plt.xlabel('p')
plt.ylabel('I(p)',rotation=0,labelpad=30)
plt.tight_layout(pad=0.15)
plt.savefig('./information.png',dpi=300)
plt.show()
plt.close()

### Produce three different pdfs and their calculate information content
### and Shannon entropy

x = np.linspace(-3,3,n)

# Dirac delta function
p_Dirac = np.zeros(len(x))
title = 'Dirac delta function'
p_Dirac[np.argmin(np.power(x,2))] = 1
I_Dirac = -1*np.log2(p_Dirac)
E_Dirac = p_Dirac*I_Dirac
E_Dirac[p_Dirac==0]=0
E_Dirac = sum(E_Dirac)
print('Entropy of Dirac delta function = ' + str(E_Dirac))

# Gaussian pdf
mu = 0
sigma = .5
p_Gaussian = mlab.normpdf(x, mu, sigma)
I_Gaussian = -1*np.log2(p_Gaussian)
E_Gaussian = p_Gaussian*I_Gaussian
E_Gaussian = sum(E_Gaussian)
print('Entropy of the Gaussian pdf = ' + str(E_Gaussian))

# Uniform pdf
p_uniform = np.ones(len(x))*1./(max(x)-min(x))
I_uniform = -1*np.log2(p_uniform)
E_uniform = p_uniform*I_uniform
E_uniform = sum(E_uniform)
print('Entropy of the uniform pdf = ' + str(E_uniform))

### Plot the comparison
f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharex=True, sharey=False, figsize=(14,5))

plot_sub(f,ax1,ax4,x,p_Dirac,I_Dirac,'Dirac delta function',label_axes=True,plot_dot=True)
plot_sub(f,ax2,ax5,x,p_Gaussian,I_Gaussian,'Gaussian pdf with $\sigma>0$')
plot_sub(f,ax3,ax6,x,p_uniform,I_uniform,'Uniform pdf')
f.text(0.19, .02, '(A)', size=BIGGER_SIZE)
f.text(0.5, .02, '(B)', size=BIGGER_SIZE)
f.text(0.81, .02, '(C)', size=BIGGER_SIZE)
plt.subplots_adjust(top=0.95, bottom=0.2, left=0.08, right=0.95, hspace=0.25,
                    wspace=0.25)
plt.savefig('./entropy_comparison',dpi=300)
plt.show()
