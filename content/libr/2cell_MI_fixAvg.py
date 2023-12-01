import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib
from scipy.optimize import root
matplotlib.rcParams['font.size']=11

C = 20 # number of discretized stimuli
# binary words
words = np.array([[0,0],[0,1],[1,0],[1,1]])
# count spikes in each word
nk = np.sum(words,1)
# product of spikes
pk = np.prod(words,1)

# tuning function
x_c = np.arange(C)+0.5
# Nx = len(x_c)
def tuning1D(x,cent,width,gain,bias=0):
    return bias + gain*np.exp(-((x - cent)/width)**2)

# compute mutual information between stimulus and response
def MI(probs): # assume flat prior
    # compute P(x) = sum_s P(x|s)P(s)
    prall = np.mean(probs,0)
    # compute P(x|s)log(P(x|s)/P(x)) for all x,s
    prrat = probs * np.log2(probs/prall)
    # sum_s P(s) sum_x P(x|s)log(P(x|s)/P(x))
    return np.sum(prrat) / len(x_c)

pox_c = np.linspace(-2,2,11)

# compute probabilities for each word
def pro(h0,  # penalization for spikes
        tun, # input tuning
        cor): # correlation
    # compute energy for each word
    probs = np.exp((words@tun).T + cor*pk - h0*nk) # exp of energy
    # normalize to sum to 1
    probs = (probs.T / (np.sum(probs,1))).T 
    return probs

def data_for_plot(width,gain,dist,penal_fix):
    tun=np.array([tuning1D(x_c,10+i*dist/2,width=width,gain=gain) for i in [-1,1]])
    MIs = [MI(pro(penal_fix[ic],tun,c)) for ic,c in enumerate(pox_c)]
    ibest,best_c = np.argmax(MIs),pox_c[np.argmax(MIs)]
    fields = pro(penal_fix[ibest],tun,best_c)@words
    avg_f = [np.mean(pro(penal_fix[ic],tun,c)@words) for ic,c in enumerate(pox_c)]
    return tun.T, MIs, fields, avg_f,best_c

# given tuning, gain, bias, width, distance, compute the right penalization for various levels of C that mantain the same avg firing rate

def pen_fix(width = 3,gain = 2,dist = 2,avg_f = 0.5):
    tun=np.array([tuning1D(x_c,10+i*dist/2,width=width,gain=gain) for i in [-1,1]])
    penaliz_fix = []
    for c in pox_c:
        def f(pen):
            return np.mean(pro(pen,tun,c)@words) - avg_f
        penaliz_fix.append(root(f,0).x[0])
    return penaliz_fix

def plot_data(tuning,MIs,fields,avg_f):
    # plot tuning
    plt.subplot(2,3,1)
    tunplot = plt.plot(tuning)
    plt.ylim([-0.05,4.05])
    plt.xticks([0,20],[0,1])
    plt.ylabel('input tuning')
    plt.xlabel('stimulus')
    # as title, puth width, bias, gain, and distance
    plt.title('$f_1, f_2$: w='+str(width)+' g='+str(gain)+' d='+str(dist))

    # plot MI for various choices of c
    plt.subplot(2,3,2)
    miplot=plt.plot(pox_c,MIs,color='darkgreen')
    best_c = pox_c[np.argmax(MIs)]
    # as title use optimal c
    plt.title('$\omega_{opt}$='+str(np.round(best_c,2)))
    vl = plt.axvline(x = best_c,color='k',alpha=0.5,ls='--')
    plt.xlabel('noise corr ($\omega$)')
    plt.ylabel(r'$MI(\mathbf{\sigma};s)$')

    # plot total resulting fields
    plt.subplot(2,3,4)
    FRplot = plt.plot(fields)
    plt.title('Marginals (Obs. Fields) @ $\omega_{opt}$')
    plt.legend(['$P(\sigma_1=1|s)$','$P(\sigma_2=1|s)$'])
    plt.ylim([-0.05,1.05])
    plt.xticks([0,20],[0,1])
    plt.xlabel('stimulus')
    plt.ylabel(r'$P(\sigma_i | s)$')

    plt.subplot(2,3,5)
    avgfplot=plt.plot(pox_c,avg_f,color='red',alpha=0.6,label='Tot. Fir')
    plt.ylabel('Avg. FR')
    plt.ylim([0.1,0.9])
    plt.xlabel('noise corr ($\omega$)')
    # as title, put avg fir for optimal omega
    plt.title('FR($\omega_{opt}$)='+str(np.round(avg_f[np.argmax(MIs)],2)))
    # plt.axvline(x = best_c,color='k',alpha=0.5,ls='--')

    plt.tight_layout()
    return tunplot, miplot, FRplot, avgfplot,vl
# initial values
width = 3
gain = 2.5
dist = 2.5
avgf = 0.5

def sliders(fig):
    # Make a horizontal slider to control the width
    ax1 = fig.add_axes([0.75, 0.8, 0.15, 0.03])
    width_slider = Slider(
        ax=ax1,
        label='Width ',
        # valfmt='%0.2f',
        valmin=1,
        valmax=5,
        valinit=width
    )
    # Make a horizontal slider to control the gain
    ax3 = fig.add_axes([0.75, 0.7, 0.15, 0.03])
    gain_slider = Slider(
        ax=ax3,
        label='Gain (SNR) ',
        valmin=0,
        valmax=5,
        valinit=gain
    )
    # Make a horizontal slider to control the distance
    ax4 = fig.add_axes([0.75, 0.6, 0.15, 0.03])
    dist_slider = Slider(
        ax=ax4,
        label='Dist ',
        valmin=0,
        valmax=5,
        valinit=dist
    )
    # Make a horizontal slider to control the avg_f
    ax5 = fig.add_axes([0.75, 0.5, 0.15, 0.03])
    avg_slider = Slider(
        ax=ax5,
        label='Avg.FR. ',
        valmin=0.2,
        valmax=0.8,
        valinit=avgf
    )
    return width_slider,gain_slider,dist_slider,avg_slider

fig = plt.figure(figsize=(12,8))
penalz = pen_fix(width,gain,dist,avgf)
tun,MIs,fields,avg_f,best_c = data_for_plot(width,gain,dist,penalz)
tunplot, miplot, FRplot, avgfplot,vl = plot_data(tun,MIs,fields,avg_f)

width_slider,gain_slider,dist_slider,avg_slider = sliders(fig)

# # The function to be called anytime a slider's value changes
def update(val):
    width = width_slider.val
    gain = gain_slider.val
    dist = dist_slider.val
    avgf = avg_slider.val
    # tunplot, miplot, FRplot, avgfplot = update_plot(width,bias,gain,dist,penalz)
    penalz = pen_fix(width,gain,dist,avgf)
    tun,MIs,fields,avg_f,best_c = data_for_plot(width,gain,dist,penalz)
    for i in range(2):
        tunplot[i].set_ydata(tun[:,i])
        # adjust title and round value to 2 decimals
        tunplot[i].axes.set_title('$f_1, f_2$: w='+str(np.round(width,2))+' g='+str(np.round(gain,2))+' d='+str(np.round(dist,2)))
        # adjust ylim 
        tunplot[i].axes.set_ylim([min(np.min(tun)*1.07,-0.05),max(4,np.max(tun)*1.07)])
        FRplot[i].set_ydata(fields[:,i])
    # adjust position of vertical line
    vl.set_xdata(best_c)
    avgfplot[0].set_ydata(avg_f)
    # adjust title
    avgfplot[0].axes.set_title('FR($\omega_{opt}$)='+str(np.round(avg_f[np.argmax(MIs)],2)))
    avgfplot[0].axes.set_ylim([0.1,0.9])
    miplot[0].set_ydata(MIs)
    # adjust title
    miplot[0].axes.set_title('$\omega_{opt}$='+str(np.round(best_c,2)))
    miplot[0].axes.set_ylim([np.min(MIs),np.max(MIs)*1.05])
    fig.canvas.draw_idle()


# # register the update function with each slider
width_slider.on_changed(update)
gain_slider.on_changed(update)
dist_slider.on_changed(update)
avg_slider.on_changed(update)

plt.show()