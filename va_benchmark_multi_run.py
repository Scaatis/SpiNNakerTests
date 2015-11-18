
# coding: utf-8
"""
This is a modification of the va_benchmark.py script to test multiple runs
and data extraction.

Simulator is hardcoded as spiNNaker, benchmark as CUBA.
"""

import os
from math import *

import pyNN.spiNNaker as sim
from pyNN.random import NumpyRNG, RandomDistribution

import matplotlib.pyplot as plt

from itertools import count, takewhile, ifilter, islice

import sys
mode = "spikes"
if len(sys.argv) > 1:
    if len(sys.argv) > 2 or sys.argv[1] == "help" or sys.argv[1] not in ("v", "spikes"):
        print "Usage: python dummy_net_multi_run.py [v|spikes|help]"
        print "v: plot voltage"
        print "spikes: plot spikes (default)"
        print "help: print this help and exit"
        if sys.argv[1] == "help":
            sys.exit(0)
        else:
            sys.exit(1)
    elif sys.argv[1] == "v":
        mode = "v"

# === Define parameters ========================================================

threads  = 1
rngseed  = 98765
parallel_safe = True

n        = 4000  # number of cells
r_ei     = 4.0   # number of excitatory cells:number of inhibitory cells
pconn    = 0.02  # connection probability
stim_dur = 50.   # (ms) duration of random stimulation
rate     = 100.  # (Hz) frequency of the random stimulation

dt       = 1.0   # (ms) simulation resolution
tstop    = 100  # (ms) total simulaton duration
delay    = 2.0

# Cell parameters
area     = 20000. # (µm²)
tau_m    = 20.    # (ms)
cm       = 1.     # (µF/cm²)
g_leak   = 5e-5   # (S/cm²)
E_leak   = -49.  # (mV)
v_thresh = -50.   # (mV)
v_reset  = -60.   # (mV)
t_refrac = 5.     # (ms) (clamped at v_reset)
v_mean   = -60.   # (mV) 'mean' membrane potential, for calculating CUBA weights
tau_exc  = 5.     # (ms)
tau_inh  = 10.    # (ms)

# Synapse parameters
Gexc = 0.27   # (nS) #Those weights should be similar to the COBA weights
Ginh = 4.5    # (nS) # but the delpolarising drift should be taken into account
Erev_exc = 0.     # (mV)
Erev_inh = -80.   # (mV)

### what is the synaptic delay???

# === Calculate derived parameters =============================================

area  = area*1e-8                     # convert to cm²
cm    = cm*area*1000                  # convert to nF
Rm    = 1e-6/(g_leak*area)            # membrane resistance in MΩ
assert tau_m == cm*Rm                 # just to check
n_exc = int(round((n*r_ei/(1+r_ei)))) # number of excitatory cells   
n_inh = n - n_exc                     # number of inhibitory cells
celltype = sim.IF_curr_exp
w_exc = 1e-3*Gexc*(Erev_exc - v_mean) # (nA) weight of excitatory synapses
w_inh = 1e-3*Ginh*(Erev_inh - v_mean) # (nA)
assert w_exc > 0; assert w_inh < 0

# === Build the network ========================================================

extra = {'threads' : threads,
         'label': 'VA'}

node_id = sim.setup(timestep=dt, min_delay=delay, max_delay=delay, **extra)

print "%s Initialising the simulator with %d thread(s)..." % (node_id, extra['threads'])
    
cell_params = {
    'tau_m'      : tau_m,    'tau_syn_E'  : tau_exc,  'tau_syn_I'  : tau_inh,
    'v_rest'     : E_leak,   'v_reset'    : v_reset,  'v_thresh'   : v_thresh,
    'cm'         : cm,       'tau_refrac' : t_refrac}

print "%s Creating cell populations..." % node_id
exc_cells = sim.Population(n_exc, celltype, cell_params,
    label="Excitatory_Cells")
inh_cells = sim.Population(n_inh, celltype, cell_params,
    label="Inhibitory_Cells")

print "%s Initialising membrane potential to random values..." % node_id
rng = NumpyRNG(seed=rngseed, parallel_safe=parallel_safe)
uniformDistr = RandomDistribution('uniform', [v_reset,v_thresh], rng=rng)
exc_cells.initialize('v', uniformDistr)
inh_cells.initialize('v', uniformDistr)

print "%s Connecting populations..." % node_id
exc_conn = sim.FixedProbabilityConnector(pconn, weights=w_exc,
    delays=delay)
inh_conn = sim.FixedProbabilityConnector(pconn, weights=w_inh,
    delays=delay)

connections={}
connections['e2e'] = sim.Projection(exc_cells, exc_cells, exc_conn,
    target='excitatory', rng=rng)
connections['e2i'] = sim.Projection(exc_cells, inh_cells, exc_conn,
    target='excitatory', rng=rng)
connections['i2e'] = sim.Projection(inh_cells, exc_cells, inh_conn,
    target='inhibitory', rng=rng)
connections['i2i'] = sim.Projection(inh_cells, inh_cells, inh_conn,
    target='inhibitory', rng=rng)


# === Setup recording ==========================================================
print "%s Setting up recording..." % node_id
if mode == "spikes":
    #exc_cells.record()
    inh_cells.record()
else:
    exc_cells.record_v()
    inh_cells.record_v()

# === Run simulation ===========================================================
print "%d Running simulation..." % node_id

plt.figure()
plt.xlabel("Time (ms)")
if mode == "spikes":
    plt.ylabel("Neuron index")
else:
    plt.ylabel("Voltage")
plt.show(block=False)

running = True
total_run_time = 0

while running:
    sim.run(tstop)
    total_run_time += tstop

    plt.xlim(max(0, total_run_time - 5*tstop), total_run_time)
    if mode == "spikes":
        plt.ylim(-1, n_inh + 1)
        all_spikes = inh_cells.getSpikes()
        spikes = list(ifilter(lambda x: x[1] > total_run_time - tstop, all_spikes))
        plt.plot([i[1] for i in spikes], [i[0] for i in spikes], "bo", markersize=3)
    else:
        plt.ylim(v_reset - 2, v_thresh + 2)
        voltages = list(ifilter(lambda x: x[0] == 5 and x[1] >= total_run_time - tstop,
            reversed(exc_cells.get_v())))
        plt.plot([i[1] for i in voltages], [i[2] for i in voltages], "b-", markersize=1)

    plt.draw()
    plt.pause(0.001)

    if not plt.get_fignums():
        running = False

sim.end()