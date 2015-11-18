
# A different version of the dummy net, for multiple runs and spike extraction

import matplotlib.pyplot as plt

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

import pyNN.spiNNaker as sim
from itertools import count, takewhile, ifilter, islice

############################## set up simulation parameters
time_to_run = 100.0 # ms, how much to run per step
dt = 1 # ms, simulation timestep

fast_spike_offset = 18.8
fast_spike_rate = 24
slow_spike_offset = 31.6
slow_spike_rate = 35.2

tau_m    = 20.    # (ms)
cm       = 1.     # (uF/cm^2)
g_leak   = 5e-5   # (S/cm^2)
E_leak = -60 # (mV)
v_thresh = -50.   # (mV)
v_reset  = -60.   # (mV)
t_refrac = 10.     # (ms) (clamped at v_reset)
tau_exc  = 5.     # (ms)
tau_inh  = 10.    # (ms)
cell_params = {
    'tau_m'      : tau_m,    'tau_syn_E'  : tau_exc,  'tau_syn_I'  : tau_inh,
    'v_rest'     : E_leak,   'v_reset'    : v_reset,  'v_thresh'   : v_thresh,
    'cm'         : cm,       'tau_refrac' : t_refrac}

############################### setup simulation
sim.setup(timestep=dt, min_delay=1.0)

# build populations
cell_type = sim.IF_cond_exp
num_of_cells = 10
layers   = 10
populations = []
for i in range(layers):
    curr_pop = sim.Population(num_of_cells, cell_type, cell_params, label="Population {0}".format(i))
    populations.append(curr_pop)
    curr_pop.initialize('v', v_reset)
if mode == "spikes":
    populations[-1].record()
else:
    populations[0].record_v()

############################### build connections between populations
syn_weight = 0.025
syn_delay = 1.0
projections = []
for i in range(layers - 1):
    curr_proj = sim.Projection(populations[i], populations[i+1], sim.AllToAllConnector(weights=syn_weight, delays=syn_delay))
    projections.append(curr_proj)

# connect spike input to the first layer
fast_injector = None
slow_injector = None
fast_injector = sim.Population(num_of_cells, sim.SpikeSourceArray, {"spike_times": [[]]*10}, label="fast_injector")
slow_injector = sim.Population(num_of_cells, sim.SpikeSourceArray, {"spike_times": [[]]*10}, label="slow_injector")
sim.Projection(fast_injector, populations[0], sim.AllToAllConnector(weights=syn_weight, delays=syn_delay))
sim.Projection(slow_injector, populations[0], sim.AllToAllConnector(weights=syn_weight, delays=syn_delay))
#fast_injector.record()

############################### run simulation
running = True
total_run_time = 0

def takewhile_alt(predicate, iterable):
    last = next(iterable)
    yield last
    while predicate(last):
        last = next(iterable)
        yield last

fast_spike_iter = (fast_spike_offset + fast_spike_rate * i for i in count())
slow_spike_iter = (slow_spike_offset + slow_spike_rate * i for i in count())
last_spike_fast = 0
last_spike_slow = 0

plt.figure()
plt.xlabel("Time (ms)")
if mode == "spikes":
    plt.ylabel("Neuron index")
else:
    plt.ylabel("Voltage")
plt.show(block=False)

while running:
    if last_spike_fast < total_run_time + time_to_run:
        fast_spikes = list(islice(fast_spike_iter, 10))
        last_spike_fast = fast_spikes[-1]
        fast_injector.set("spike_times", [fast_spikes] + [[]] * 9)

    if last_spike_slow < total_run_time + time_to_run:
        slow_spikes = list(islice(slow_spike_iter, 10))
        last_spike_slow = slow_spikes[-1]
        slow_injector.set("spike_times", [slow_spikes] + [[]] * 9)

    sim.run(time_to_run)
    total_run_time += time_to_run

    plt.xlim(max(0, total_run_time - 5*time_to_run), total_run_time)
    if mode == "spikes":
        plt.ylim(-1, 101)
        all_spikes = populations[-1].getSpikes()
        print "Total spikes %d" % len(all_spikes)
        spikes = list(takewhile(lambda x: x[1] > total_run_time - time_to_run, all_spikes))
        plt.plot([i[1] for i in spikes], [i[0] for i in spikes], ".", markersize=2)
    else:
        plt.ylim(v_reset - 5, v_thresh + 5)
        voltages = list(ifilter(lambda x: x[0] == 1 and x[1] >= total_run_time - time_to_run,
            reversed(populations[0].get_v())))
        plt.plot([i[1] for i in voltages], [i[2] for i in voltages], "b-", markersize=1)

    plt.draw()
    plt.pause(0.001)

    if not plt.get_fignums():
        running = False

sim.end()