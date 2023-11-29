import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
import matplotlib
from matplotlib.widgets import Slider
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.size']=12

C = 20 # discretization of [0,1]: effective number of stimuli
# binary words
words = np.array([[0,0],[0,1],[1,0],[1,1]])
# count spikes in each word
nk = np.sum(words,1)
# product of spikes
pk = np.prod(words,1)
# tuning function
x_c = np.arange(C)+0.5
def tuning1D(x,cent,width,gain,bias=0):
    return bias + gain*np.exp(-((x - cent)/width)**2)
# compute P(x|s) for all x,s
def pro(tun, # input tuning
        cor): # correlation
    # compute energy for each word
    probs = np.exp((words@tun).T + cor*pk) # exp of energy
    # normalize to sum to 1
    probs = (probs.T / (np.sum(probs,1))).T 
    return probs

# compute mutual information between stimulus and response
def MI(probs): # !assume flat prior!
    # compute P(x) = sum_s P(x|s)P(s)
    prall = np.mean(probs,0)
    # compute P(x|s)log(P(x|s)/P(x)) for all x,s
    prrat = probs * np.log2(probs/prall)
    # sum_s P(s) sum_x P(x|s)log(P(x|s)/P(x))
    return np.sum(prrat) / len(x_c)

# get tuning and nc, and compute the fields
def get_tun(width = 3,bias = 0,gain = 2,dist = 2,cor = 0.):
    tun=np.array([tuning1D(x_c,10+i*dist/2,width=width,gain=gain,bias=bias) for i in [-1,1]])
    fields = pro(tun,cor)@words
    return tun,fields.T
# given effective fields, and correlation, find the corresponding inputs
def find_tun(fields,cor):
    def f(tun):
        return ((pro(tun.reshape(fields.shape),cor)@words).T - fields).flatten()
    return root(f,fields.flatten()).x.reshape(fields.shape)

## to plot an example
# tun,fields = get_tun(width = 3,bias = 0,gain = 2,dist = 2,cor = 0.5)

# plt.figure(figsize=(4,3))
# plt.plot(fields[0],label='$P(\sigma_1=1|s)$')
# plt.plot(fields[1],label='$P(\sigma_2=1|s)$')
# plt.legend(loc=(0.7,0.7))
# plt.xticks([0,19],[0,1])
# plt.xlabel('stimulus')
# plt.ylabel(r'$P(\sigma|s)$  ("firing rate")')
# plt.title('Observed Fields (i.e. marginal stats)')

## plotting
def plot_data(tuning,fields,mi):
    # plot tuning
    plt.subplot(1,3,1)
    tunplot = plt.plot(tuning.T)
    plt.xticks([0,20],[0,1])
    plt.ylabel('input tuning')
    plt.xlabel('stimulus')
    plt.ylim([-1,3])
    plt.title('$\omega$=0')

    # plot total resulting fields
    plt.subplot(1,3,2)
    FRplot = plt.plot(fields.T)
    plt.xticks([0,20],[0,1])
    plt.xlabel('stimulus')
    plt.ylabel(r'$P(x_i | s)$')
    plt.title('MI='+str(np.round(mi,2)))

    plt.tight_layout()
    return tunplot, FRplot



def sliders(fig):
    # Make a horizontal slider to control the width
    ax1 = fig.add_axes([0.8, 0.6, 0.1, 0.03])
    nc_slider = Slider(
        ax=ax1,
        label='Noise Corr $\omega$',
        # valfmt='%0.2f',
        valmin=-1,
        valmax=1,
        valinit=0
    )
    return nc_slider


fig = plt.figure(figsize=(8,3))
tun,fields = get_tun(width,bias,gain,dist,cor = 0.)
# mutual information
mi = MI(pro(tun,0))
tunplot, FRplot = plot_data(tun,fields,mi)

nc_slider = sliders(fig)

# # The function to be called anytime a slider's value changes
def update(val):
    nc = nc_slider.val
    tun2 = find_tun(fields,nc)
    for i in range(2):
        tunplot[i].set_ydata(tun2[i])
        tunplot[i].axes.set_title('$\omega=$'+str(np.round(nc,2)))
        FRplot[i].axes.set_title('MI='+str(np.round(MI(pro(tun2,nc)),2)))
    fig.canvas.draw_idle()


# # register the update function with each slider
nc_slider.on_changed(update)

plt.suptitle('Fix. marginal stats, vary $f$ as a funct of $\omega$',y=1.0)
plt.tight_layout()
plt.show()