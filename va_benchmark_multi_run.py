
# coding: utf-8
"""
This is a modification of the va_benchmark.py script to test multiple runs
and data extraction.

Simulator is hardcoded as spiNNaker, benchmark as CUBA.
"""

import os
import socket
from math import *

import pyNN.spiNNaker as Frontend
from pyNN.utility import Timer
from pyNN.random import NumpyRNG, RandomDistribution

timer = Timer()

# === Define parameters ========================================================

threads  = 1
rngseed  = 98765
parallel_safe = True

n        = 4000  # number of cells
r_ei     = 4.0   # number of excitatory cells:number of inhibitory cells
pconn    = 0.02  # connection probability
stim_dur = 50.   # (ms) duration of random stimulation
rate     = 100.  # (Hz) frequency of the random stimulation

dt       = 0.1   # (ms) simulation resolution
tstop    = 30  # (ms) total simulaton duration
tstep    = 10    # (ms) simulation step duration
delay    = 0.2

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
celltype = Frontend.IF_curr_exp
w_exc = 1e-3*Gexc*(Erev_exc - v_mean) # (nA) weight of excitatory synapses
w_inh = 1e-3*Ginh*(Erev_inh - v_mean) # (nA)
assert w_exc > 0; assert w_inh < 0

# === Build the network ========================================================

extra = {'threads' : threads,
         'label': 'VA'}

node_id = Frontend.setup(timestep=dt, min_delay=delay, max_delay=delay, **extra)

host_name = socket.gethostname()
print "Host #%d is on %s" % (node_id+1, host_name)

print "%s Initialising the simulator with %d thread(s)..." % (node_id, extra['threads'])
    
cell_params = {
    'tau_m'      : tau_m,    'tau_syn_E'  : tau_exc,  'tau_syn_I'  : tau_inh,
    'v_rest'     : E_leak,   'v_reset'    : v_reset,  'v_thresh'   : v_thresh,
    'cm'         : cm,       'tau_refrac' : t_refrac}
    
timer.start()

print "%s Creating cell populations..." % node_id
exc_cells = Frontend.Population(n_exc, celltype, cell_params,
	label="Excitatory_Cells")
inh_cells = Frontend.Population(n_inh, celltype, cell_params,
	label="Inhibitory_Cells")

print "%s Initialising membrane potential to random values..." % node_id
rng = NumpyRNG(seed=rngseed, parallel_safe=parallel_safe)
uniformDistr = RandomDistribution('uniform', [v_reset,v_thresh], rng=rng)
exc_cells.initialize('v', uniformDistr)
inh_cells.initialize('v', uniformDistr)

print "%s Connecting populations..." % node_id
exc_conn = Frontend.FixedProbabilityConnector(pconn, weights=w_exc,
	delays=delay)
inh_conn = Frontend.FixedProbabilityConnector(pconn, weights=w_inh,
	delays=delay)

connections={}
connections['e2e'] = Frontend.Projection(exc_cells, exc_cells, exc_conn,
	target='excitatory', rng=rng)
connections['e2i'] = Frontend.Projection(exc_cells, inh_cells, exc_conn,
	target='excitatory', rng=rng)
connections['i2e'] = Frontend.Projection(inh_cells, exc_cells, inh_conn,
	target='inhibitory', rng=rng)
connections['i2i'] = Frontend.Projection(inh_cells, inh_cells, inh_conn,
	target='inhibitory', rng=rng)


# === Setup recording ==========================================================
print "%s Setting up recording..." % node_id
exc_cells.record()
inh_cells.record()
#exc_cells[[0, 1]].record_v()

buildCPUTime = timer.diff()

# === Run simulation ===========================================================
print "%d Running simulation..." % node_id

for i in range(tstop // tstep):
	print "Starting run %d" % (i+1)
	Frontend.run(tstep)

	simCPUTime = timer.diff()

	exc_spikes = exc_cells.getSpikes()
	inh_spikes = inh_cells.getSpikes()

	writeCPUTime = timer.diff()

	print "Simulating took %g ms" % simCPUTime
	print "Reading spikes took %g ms" % writeCPUTime
	print "A total of %d spikes was read" % (len(exc_spikes) + len(inh_spikes))

Frontend.end()