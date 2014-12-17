"""
The spike queue class stores future synaptic events
produced by a given presynaptic neuron group (or postsynaptic for backward
propagation in STDP).
"""
import bisect

import numpy as np

from brian2.memory.dynamicarray import DynamicArray1D
from brian2.utils.logger import get_logger

__all__=['SpikeQueue']

logger = get_logger(__name__)

INITIAL_MAXSPIKESPER_DT = 1


class SpikeQueue(object):
    '''
    Data structure saving the spikes and taking care of delays.
    
    Parameters
    ----------

    synapses : list of ndarray 
        A list of synapses (synapses[i]=array of synapse indices for neuron i).
    delays : ndarray
        An array of delays (delays[k]=delay of synapse k).
    dt : `Quantity`
        The timestep of the source group
    max_delay : `Quantity`, optional
        The maximum delay (in second) of synaptic events. At run time, the
        structure is resized to the maximum delay in `delays`, and thus
        the `max_delay` should only be specified if delays can change
        during the simulation (in which case offsets should not be
        precomputed).
    precompute_offsets : bool, optional
        A flag to precompute offsets. Defaults to ``True``, i.e. offsets (an
        internal array derived from `delays`, used to insert events in the data
        structure, see below) are precomputed for all neurons when the object
        is prepared with the `compress` method. This usually results in a speed
        up but takes memory, which is why it can be disabled.

    Notes
    -----

    **Data structure** 
    
    A spike queue is implemented as a 2D array `X` that is circular in the time
    direction (rows) and dynamic in the events direction (columns). The
    row index corresponding to the current timestep is `currentime`.
    Each element contains the target synapse index.    
    
    **Offsets**
    
    Offsets are used to solve the problem of inserting multiple synaptic events
    with the same delay. This is difficult to vectorise. If there are n synaptic
    events with the same delay, these events are given an offset between 0 and
    n-1, corresponding to their relative position in the data structure.
    They can be either precalculated (faster), or determined at run time
    (saves memory). Note that if they are determined at run time, then it is
    possible to also vectorise over presynaptic spikes.
    '''
    
    def __init__(self, source_start, source_end, dtype=np.int32,
                 precompute_offsets=True):
        #: Whether the offsets should be precomputed
        self._precompute_offsets = precompute_offsets

        #: The start of the source indices (for subgroups)
        self._source_start = source_start

        #: The end of the source indices (for subgroups)
        self._source_end = source_end

        self.dtype=dtype
        self.X = np.zeros((1,1), dtype=dtype) # target synapses
        self.X_flat = self.X.reshape(1, )
        #: The current time (in time steps)
        self.currenttime = 0
        #: number of events in each time step
        self.n = np.zeros(1, dtype=int)
        #: precalculated offsets
        self._offsets = None

        #: The dt used for storing the spikes (will be set in `prepare`)
        self._dt = None

        #: Storage for the store/restore mechanism
        self._stored_spikes = {}

    def prepare(self, delays, dt, synapse_sources):
        '''
        Prepare the data structure and pre-compute offsets.
        This is called every time the network is run. The size of the
        of the data structure (number of rows) is adjusted to fit the maximum
        delay in `delays`, if necessary. Offsets are calculated, unless
        the option `precompute_offsets` is set to ``False``. A flag is set if
        delays are homogeneous, in which case insertion will use a faster method
        implemented in `insert_homogeneous`.        
        '''
        n_synapses = len(synapse_sources)

        if self._dt is not None:
            # store the current spikes
            spikes = self._extract_spikes()
            # adapt the spikes to the new dt if it changed
            if self._dt != dt:
                spiketimes = spikes[:, 0] * self._dt
                spikes[:, 0] = np.round(spiketimes / dt).astype(np.int)
        else:
            spikes = None

        if len(delays):
            delays = np.array(np.round(delays / dt)).astype(np.int)
            max_delays = max(delays)
            min_delays = min(delays)
        else:
            max_delays = min_delays = 0

        self._delays = delays

        # Prepare the data structure used in propagation
        synapse_sources = synapse_sources[:]        
        ss = np.ravel(synapse_sources)
        # mergesort to retain relative order, keeps the output lists in sorted order
        I = np.argsort(ss, kind='mergesort')
        ss_sorted = ss[I]
        splitinds = np.searchsorted(ss_sorted, np.arange(self._source_start, self. _source_end+1))
        self._neurons_to_synapses = [I[splitinds[j]:splitinds[j+1]] for j in xrange(len(splitinds)-1)]
        max_events = max(map(len, self._neurons_to_synapses))

        n_steps = max_delays + 1
        
        # Adjust the maximum delay and number of events per timestep if necessary
        # Check if delays are homogeneous
        self._homogeneous = (max_delays == min_delays)

        # Resize
        if (n_steps > self.X.shape[0]) or (max_events > self.X.shape[1]): # Resize
            # Choose max_delay if is is larger than the maximum delay
            n_steps = max(n_steps, self.X.shape[0])
            max_events = max(max_events, self.X.shape[1])
            self.X = np.zeros((n_steps, max_events), dtype=self.dtype) # target synapses
            self.X_flat = self.X.reshape(n_steps*max_events,)
            self.n = np.zeros(n_steps, dtype=int) # number of events in each time step

        # Precompute offsets
        if self._precompute_offsets:
            self._do_precompute_offsets(n_synapses)

        # Re-insert the spikes into the data structure
        if spikes is not None:
            self._store_spikes(spikes)

        self._dt = dt

    def _extract_spikes(self):
        '''
        Get all the stored spikes

        Returns
        -------
        spikes : ndarray
            A 2d array with two columns, where each row describes a spike.
            The first column gives the time (as integer time steps) and the
            second column gives the index of the target synapse.
        '''
        spikes = np.zeros((np.sum(self.n), 2))
        counter = 0
        for idx, n in enumerate(self.n):
            t = (idx - self.currenttime) % len(self.n)
            for target in self.X[idx, :n]:
                spikes[counter, :] = np.array([t, target])
                counter += 1
        return spikes

    def _store_spikes(self, spikes):
        '''
        Store a list of spikes at the given positions after clearing all
        spikes in the queue.

        Parameters
        ----------
        spikes : ndarray
            A 2d array with two columns, where each row describes a spike.
            The first column gives the time (as integer time steps) and the
            second column gives the index of the target synapse.

        '''
        # Clear all spikes
        self.n[:] = 0
        for t, target in spikes:
            row_idx = (t + self.currenttime) % len(self.n)
            self.X[row_idx, self.n[row_idx]] = target
            self.n[row_idx] += 1

    def _store(self, name='default'):
        self._stored_spikes[name] = self._extract_spikes()

    def _restore(self, name='default'):
        if name in self._stored_spikes:
            self._store_spikes(self._stored_spikes[name])
        else:
            # It is possible that _store was called in `SynapticPathway`, before
            # the `SpikeQueue` was created. In that case, delete all spikes in
            # the queue
            self._store_spikes(np.empty((0, 2)))

    ################################ SPIKE QUEUE DATASTRUCTURE ################
    def advance(self):
        '''
        Advances by one timestep
        '''
        self.n[self.currenttime]=0 # erase
        self.currenttime=(self.currenttime+1) % len(self.n)
        
    def peek(self):
        '''
        Returns the all the synaptic events corresponding to the current time,
        as an array of synapse indexes.
        '''      
        return self.X[self.currenttime,:self.n[self.currenttime]]    
    
    def push(self, sources):
        '''
        Push spikes to the queue.

        Parameters
        ----------
        sources : ndarray of int
            The indices of the neurons that spiked.
        '''
        if len(sources) and len(self._delays):
            start = self._source_start
            stop = self._source_end
            if start > 0:
                start_idx = bisect.bisect_left(sources, start)
            else:
                start_idx = 0
            if stop <= sources[-1]:
                stop_idx = bisect.bisect_left(sources, stop, lo=start_idx)
            else:
                stop_idx = len(self._neurons_to_synapses)
            sources = sources[start_idx:stop_idx]
            if len(sources)==0:
                return
            synapse_indices = self._neurons_to_synapses
            indices = np.concatenate([synapse_indices[source - start]
                                      for source in sources]).astype(np.int32)
            if self._homogeneous:  # homogeneous delays
                self._insert_homogeneous(self._delays[0], indices)
            elif self._offsets is None:  # vectorise over synaptic events
                # there are no precomputed offsets, this is the case
                # (in particular) when there are dynamic delays
                self._insert(self._delays[indices], indices)
            else: # offsets are precomputed
                self._insert(self._delays[indices], indices, self._offsets[indices])
                # Note: the trick can only work if offsets are ordered in the right way

    def _do_precompute_offsets(self, n_synapses):
        '''
        Precompute all offsets corresponding to delays. This assumes that
        delays will not change during the simulation.
        '''
        if len(self._delays) == 1 and n_synapses != 1:
            # We have a scalar delay value
            delays = self._delays.repeat(n_synapses)
        else:
            delays = self._delays
        self._offsets = np.zeros_like(delays)
        index = 0
        for targets in self._neurons_to_synapses:
            target_delays = delays[targets]
            self._offsets[index:index+len(target_delays)] = self._calc_offsets(target_delays)
            index += len(target_delays)

    def _calc_offsets(self, delay):
        '''
        Calculates offsets corresponding to a delay array.
        If there n identical delays, there are given offsets between
        0 and n-1.
        Example:

            [7,5,7,3,7,5] -> [0,0,1,0,2,1]

        The code is complex because tricks are needed for vectorisation.
        '''
        # We use merge sort because it preserves the input order of equal
        # elements in the sorted output
        I = np.argsort(delay, kind='mergesort')
        xs = delay[I]
        J = xs[1:]!=xs[:-1]
        A = np.hstack((0, np.cumsum(J)))
        B = np.hstack((0, np.cumsum(-J)))
        BJ = np.hstack((0, B[J]))
        ei = B-BJ[A]
        ofs = np.zeros_like(delay)
        ofs[I] = np.array(ei, dtype=ofs.dtype) # maybe types should be signed?
        return ofs

    def _insert(self, delay, target, offset=None):
        '''
        Vectorised insertion of spike events.

        Parameters
        ----------

        delay : ndarray
            Delays in timesteps.

        target : ndarray
            Target synaptic indices.

        offset : ndarray
            Offsets within timestep. If unspecified, they are calculated
            from the delay array.
        '''
        delay = np.array(delay, dtype=int)

        if offset is None:
            offset = self._calc_offsets(delay)

        # Calculate row indices in the data structure
        timesteps = (self.currenttime + delay) % len(self.n)
        # (Over)estimate the number of events to be stored, to resize the array
        # It's an overestimation for the current time, but I believe a good one
        # for future events
        m = max(self.n) + len(target)
        if (m >= self.X.shape[1]): # overflow
            self._resize(m+1)

        self.X_flat[timesteps*self.X.shape[1]+offset+self.n[timesteps]] = target
        self.n[timesteps] += offset+1 # that's a trick (to update stack size)

    def _insert_homogeneous(self, delay, target):
        '''
        Inserts events at a fixed delay.

        Parameters
        ----------
        delay : int
            Delay in timesteps.

        target : ndarray
            Target synaptic indices.
        '''
        timestep = (self.currenttime + delay) % len(self.n)
        nevents = len(target)
        m = self.n[timestep]+nevents+1 # If overflow, then at least one self.n is bigger than the size
        if (m >= self.X.shape[1]):
            self._resize(m + 1)  # was m previously (not enough)
        k = timestep*self.X.shape[1] + self.n[timestep]
        self.X_flat[k:k+nevents] = target
        self.n[timestep] += nevents

    def _resize(self, maxevents):
        '''
        Resizes the underlying data structure (number of columns = spikes per
        dt).

        Parameters
        ----------
        maxevents : int
            The new number of columns. It will be rounded to the closest power
            of 2.
        '''
        # old and new sizes
        old_maxevents = self.X.shape[1]
        new_maxevents = int(2**np.ceil(np.log2(maxevents))) # maybe 2 is too large
        # new array
        newX = np.zeros((self.X.shape[0], new_maxevents), dtype = self.X.dtype)
        newX[:, :old_maxevents] = self.X[:, :old_maxevents] # copy old data

        self.X = newX
        self.X_flat = self.X.reshape(self.X.shape[0]*new_maxevents,)
