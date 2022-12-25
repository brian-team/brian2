#!/usr/bin/env python
# coding: utf-8

"""
Reproduction of Fig. 2 (b and c) of Spreizer et al. 2019 in Brian: 

Spreizer S, Aertsen A, Kumar A. From space to time: Spatial inhomogeneities lead to the emergence of spatiotemporal sequences in spiking neuronal networks. PLOS Computational Biology. 2019;15(10):e1007432. doi:10.1371/journal.pcbi.1007432

Requirements:

- Brian
- numpy 
- matplotlib
- noise: For setting up the Perlin landscape. Otherwise, the landscapes
  for different network sizes (scaled down by a factor of 1 to 5) can be
  downloaded from the following:

  https://github.com/arashgmn/arashgmn.github.io/tree/master/assets/data/perlin-for-brian

This code was written by Arash Golmohammadi, 2022.
"""

import numpy as np
import brian2 as b2
from brian2.units import ms, pA, mV

import matplotlib.pyplot as plt
from matplotlib import animation

b2.seed(8)

def round_to_even(gs, scaler):
    """
    rounds the network size to an even number.
    """
    rounded = round(gs/scaler)
    if rounded%2:
        rounded+=1
    return int(rounded)

# ---------------- CONFIGURATIONS ---------------- # 

# NETWORK TOPOLOGY
SCALAR = 3 # For faster omputation I recommand running with SCALAR 2 or 3
GRID_SIZE = round_to_even(100, SCALAR)
NCONN = round_to_even(1000, SCALAR**2)

# CELL CONFIGS
THR, REF, RESET = -55*mV, 2*ms, -70*mV
        
# PROFILE 
THETA, KAPPA, GAP = 3/SCALAR, 4, 3

# ANISOTROPY
R, SCALE = 1., 4 # perlin noise
PHI_H = np.pi/6 # homog. angle

# WARMUP
DUR_WU, STD_WU = 500*ms, 500*pA

# BACKGROUND
MU, SIGMA = 700*pA, 100* pA



EQ_NRN = """
    dv/dt = (E-v)/tau_m + (noise + I_syn)/C : volt (unless refractory)
    I_syn :  amp
    noise  = mu + sigma*sqrt(noise_dt)*xi_pop: amp
    
    mu               : amp (shared)
    sigma            : amp (shared)
    noise_dt = 1*ms  : second (shared)
    C = 250*pF       : farad (shared)
    tau_m = 10*ms      : second (shared)
    E = -70*mV       : volt (shared)
"""

EQ_SYN = """
    dg/dt = (-g+h) / tau_s : 1 (clock-driven)
    dh/dt = -h / tau_s     : 1 (clock-driven)
    I_syn_post = J*g  : amp  (summed)
    J = -{}*10*pA           : amp (shared)
    tau_s = 5*ms           : second (shared)
""".format(SCALAR**2)

ON_PRE = "h += exp(1)" # to match NEST's normalization


# ---------------- UTILITIES ---------------- # 

def coord2idx(coords, pop):
    """Converts array of coordinates to indices."""
    coords = np.asarray(coords).reshape(-1,2)
    idxs = coords[:,1]*GRID_SIZE + coords[:,0]
    return idxs

def idx2coord(idxs, pop):
    """Converts array of indices to coordinates"""
    idxs = np.asarray(idxs)
    y,x = np.divmod(idxs, GRID_SIZE)
    coords = np.array([x,y]).T
    return coords

def make_periodic(x,y, grid_size= GRID_SIZE):
    x = x % GRID_SIZE
    y = y % GRID_SIZE
    return x,y 


# ---------------- LANDSCAPE GENERATION ---------------- # 

def landscape(anisotropy='perlin'):
    """Generates an angular bias for each neuron in the 
    network such that no angle is more likely than the 
    other. Yet, nearby bias angles may be correleted.
    
    Note that this is a flattened array."""
    
    if anisotropy=='perlin': # requires noise package
        from noise import pnoise2 as perlin

        x = y = np.linspace(0, SCALE, GRID_SIZE)
        phis = [[perlin(i, j, repeatx=SCALE, repeaty=SCALE) for j in y] for i in x]
        phis = np.concatenate(phis)
        phis = balance_landscape(phis)
        phis -= phis.min()
        phis *= 2*np.pi/phis.max()
    
    elif anisotropy=='homog':
        phis = np.ones(GRID_SIZE**2) * PHI_H
    
    else: # both random and symmetric have the same landscape
        phis = np.random.uniform(0,2*np.pi, size=GRID_SIZE**2)
    
    return phis

def balance_landscape(array):
    """Equalizes the histogram."""
    
    sorted_idx = np.argsort(array)
    gs = int(np.sqrt(len(array)))
    max_val = gs * 2
    idx = int(len(array) // max_val)
    for i, val in enumerate(range(max_val)):
        array[sorted_idx[i * idx:(i + 1) * idx]] = val
    
    return (array- gs) / gs


# ---------------- VISUALIZATION ---------------- # 

def get_cax(ax):
    """
    Returns a nice colorbar.
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    
    cax = inset_axes(ax, width="5%", height="100%", 
                     loc='lower left', bbox_to_anchor=(1.05, 0., 1, 1),
                     bbox_transform= ax.transAxes, borderpad=0)
    
    return cax


def visualize(net, mon, name='perlin'):

    idxs, ts = mon.it
    
    def plot_firing_rate_disto():
        """plots distribution of firing rates over the simulation"""
        
        fig = plt.figure(figsize=(10,5))
        ax = plt.gca()
        
        T = np.max(ts) - np.min(ts)
        _, rates = np.unique(idxs, return_counts=True)
        rates = rates*1./T
        ax.hist(rates, bins=50, density=True,)
        ax.set_xlabel('Firing rate [Hz]')
        ax.set_ylabel('Probability density')
        ax.set_yscale('log')
        ax.set_title('Population I')
        
        plt.savefig(name+'_rate.png',bbox_inches='tight', dpi=200)
        plt.close()

        
    def plot_in_out_deg():
        """plots distribution as well as the heat map of in- and
        out-degree from each neuron."""
        
        syn = net['syn_II']
        
        src = syn.source
        trg = syn.target
        
        in_deg = syn.N_incoming_post.reshape((GRID_SIZE, GRID_SIZE))
        out_deg = syn.N_outgoing_pre.reshape((GRID_SIZE, GRID_SIZE))
        
        fig, axs = plt.subplots(2,2, figsize=(10,5), constrained_layout=True,
                                gridspec_kw={'height_ratios':[1,.5]})
        
        # field map & distribution of in-degrees
        m = axs[0, 0].pcolormesh(in_deg, shading='flat')
        plt.colorbar(m, cax= get_cax(axs[0,0]))
        axs[1 , 0].hist(syn.N_incoming_post, bins=50, density=True)
        
        # field map & distribution of out-degrees
        m = axs[0, 1].pcolormesh(out_deg, shading='flat')
        plt.colorbar(m, cax= get_cax(axs[0,1]))
        axs[1 , 1].hist(syn.N_outgoing_pre, bins=50, density=True)
        
        
        axs[1,0].set_ylabel('Probability density')
        axs[1,0].set_xlabel('In-degree '+src.name+r'$\to$'+trg.name)
        axs[1,1].set_xlabel('out-degree '+src.name+r'$\to$'+trg.name)
        
        for ax in axs[0,:]:
            ax.set_aspect('equal')
            ax.get_yaxis().set_ticks([])
        
        plt.savefig(name+'_degrees.png',bbox_inches='tight', dpi=200)
        plt.close()

        
    def animator(fig, axs, imgs, vals, ts_bins=[]):
        """used for making an activity animation"""
        n_frames = len(vals[0]) # total number of frames

        def animate(frame_id):
            if len(imgs)>1:
                for pop_idx in range(len(imgs)):
                    imgs[pop_idx].set_array(vals[pop_idx][frame_id])
            else:
                imgs[0].set_array(vals[0][frame_id])

            if len(ts_bins) > 0:
                fig.suptitle('%s' % ts_bins[frame_id])
            else:
                fig.suptitle('%s' % frame_id)

            return *imgs,

        # Call the animator. 
        anim = animation.FuncAnimation(fig, animate, frames=n_frames,
                                       interval=50, blit=True)

        return anim


    def plot_animation(ss_dur=25, fps=10,):
        """Plots neuronal activity as an animation"""
        
        fig, axs = plt.subplots(1, 1)
        axs = [axs]

        ts_bins = np.arange(0, net.t/ms + 1, ss_dur) * ms

        field_imgs = []
        field_vals = []

        idxs, ts = mon.it
        h = np.histogram2d(ts, idxs, bins=[ts_bins, range(GRID_SIZE**2 + 1)])[0]
        field_val = h.reshape(-1, GRID_SIZE, GRID_SIZE)
        field_img = axs[0].imshow(field_val[0], 
                                    vmin=0, vmax=np.max(field_val), 
                                    origin='lower'  
                                    )
        axs[0].set_title('Population I')

        field_vals.append(field_val)
        field_imgs.append(field_img)

        anim = animator(fig, axs, field_imgs, field_vals, ts_bins)

        writergif = animation.PillowWriter(fps=fps) 
        anim.save(name+'.gif', writer=writergif)
        plt.close()
        
    
    plot_firing_rate_disto()
    plot_in_out_deg()
    plot_animation()


# ---------------- NETWORK SETUP ---------------- # 

def setup_net():
    I = b2.NeuronGroup(N = GRID_SIZE**2, 
                     name = 'I', 
                     model = EQ_NRN, 
                     refractory = 5*ms,
                     threshold='v>{}*mV'.format(THR/mV),
                     reset='v={}*mV'.format(RESET/mV),
                     method='euler',
                     )
    I.mu = MU
    I.sigma = SIGMA
    
    syn_II = b2.Synapses(I, I, 
                         model= EQ_SYN, 
                         on_pre= ON_PRE,
                         method='exact',
                         name = 'syn_II'
                         )
    
    
    for s_idx in range(len(I)):
        
        rel_coords = draw_post(s_idx, phis[s_idx])
        s_coord = idx2coord(s_idx, I) # check shape
        x,y = (rel_coords + s_coord).T
        x,y = make_periodic(x,y)
        t_coords = np.array([x,y]).T.astype(int)
        t_idxs = coord2idx(t_coords, I)
        
        syn_II.connect(i = s_idx, j = t_idxs)
    
    return I, syn_II


def draw_post(idx, phi):
    
    alpha = np.random.uniform(-np.pi, np.pi, NCONN)
    radius = np.concatenate([
             - np.random.gamma(shape= KAPPA, scale= THETA, 
                              size= int(NCONN // 2)
                              ),
             + np.random.gamma(shape= KAPPA, scale= THETA, 
                               size= NCONN -int(NCONN // 2))
            ])
    
    radius[radius< 0] -= GAP
    radius[radius>=0] += GAP
    
    x, y = radius*np.cos(alpha), radius*np.sin(alpha) 
    x += R*np.cos(phi)
    y += R*np.sin(phi)
    
    eps = 1e-3
    self_link = x**2 + y**2 < eps**2 
    x = x[~self_link]
    y = y[~self_link]
    
    coords = np.array([x,y]).T
    return np.round(coords).astype(int)

def warmup(net):
    net['I'].mu = 0*pA
    net['I'].sigma = STD_WU
    
    net.run(DUR_WU/2)
    net['I'].mu = MU
    net['I'].sigma = SIGMA
    net.run(DUR_WU/2)
    
    
def simulate(dur = 3000*ms):
    
    b2.start_scope()
    I, syn_II = setup_net()
    mon = b2.SpikeMonitor(I, record=True)
    
    net = b2.Network()
    net.add(I)
    net.add(syn_II)
    net.add(mon)
    
    warmup(net)
    net.run(3000*ms)
        
    return net, mon


# ---------------- MAIN ---------------- # 
for lscp in ['symmetric', 'random', 'homog', 'perlin']:
    print('Simulating {} landscape.'.format(lscp))
    if lscp=='symmetric': # no displacement
        R = 0.

    phis = landscape(lscp)
    net, mon = simulate()
    visualize(net, mon, lscp)
    
    if lscp=='symmetric': # no displacement
        R = 1.

print('Results are saved in the current directory.')




