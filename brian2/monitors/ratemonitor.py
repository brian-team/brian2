"""
Module defining `PopulationRateMonitor` and the abstract `RateMonitor` class.
"""

from abc import ABC, abstractmethod

import numpy as np

from brian2.core.functions import timestep
from brian2.core.variables import Variables
from brian2.groups.group import CodeRunner, Group
from brian2.units.allunits import hertz, second
from brian2.units.fundamentalunits import Quantity, check_units
from brian2.utils.logger import get_logger

__all__ = ["PopulationRateMonitor", "RateMonitor"]


logger = get_logger(__name__)


class RateMonitor(CodeRunner, Group, ABC):
    """
    Abstract base class for monitors that record rates.
    """

    @abstractmethod
    @check_units(bin_size=second)
    def binned_rate(self, bin_size):
        """
        Return the rate calculated in bins of a certain size.

        Parameters
        -------------
        bin_size : `Quantity`
            The size of the bins in seconds. Should be a multiple of dt.

        Returns
        -------
        bins : `Quantity`
            The start time of the bins.
        binned_values : `Quantity`
            The binned values. For EventMonitor subclasses, this is a 2D array
            with shape (neurons, bins). For PopulationRateMonitor, this is a 1D array.
        Notes
        -----
        The returned bin times represent the **start** of each bin interval, not the center.
        This is consistent with how Brian2 records spike times and other temporal data.
        For example, a spike recorded at time `t` occurred during the interval `[t, t+dt)`.

        For plotting purposes, especially with larger bin sizes, you may want to use bin
        centers instead of bin starts for a more intuitive visualization. You can easily
        calculate the bin centers by adding half the bin size::

            >> bins, rates = monitor.binned_rate(10*ms)
            >> bin_centers = bins + 10*ms / 2
            >> plt.plot(bin_centers, rates)

        This adjustment is particularly helpful when the bins are large relative to the
        time scale of interest, as it better represents where the rate measurement applies
        within each time window.
        """
        raise NotImplementedError()

    @check_units(width=second)
    def smooth_rate(self, window="gaussian", width=None):
        """
        Returns a smoothed out version of the firing rate(s).

        Parameters
        ----------
        window : str, ndarray
            The window to use for smoothing. Can be a string to chose a
            predefined window(`flat` for a rectangular, and `gaussian`
            for a Gaussian-shaped window).

            In this case the width of the window
            is determined by the `width` argument. Note that for the Gaussian
            window, the `width` parameter specifies the standard deviation of
            the Gaussian, the width of the actual window is `4*width + dt`
            (rounded to the nearest dt). For the flat window, the width is
            rounded to the nearest odd multiple of dt to avoid shifting the rate
            in time.
            Alternatively, an arbitrary window can be given as a numpy array
            (with an odd number of elements). In this case, the width in units
            of time depends on the ``dt`` of the simulation, and no `width`
            argument can be specified. The given window will be automatically
            normalized to a sum of 1.
        width : `Quantity`, optional
            The width of the ``window`` in seconds (for a predefined window).

        Returns
        -------
        rate : `Quantity`
            The smoothed firing rate(s) in Hz. For EventMonitor subclasses,
            this returns a 2D array with shape (neurons, time_bins).
            For PopulationRateMonitor, this returns a 1D array.
            Note that the rates are smoothed at the original time step resolution (dt), not re-binned.
            The length of the returned array equals the number of recorded time steps and
            can be plotted against the original time values (e.g., self.t).

        Warnings
        --------
        This method will give incorrect results if the monitor has recorded values with
        varying ``dt`` values.
        """
        if width is None and isinstance(window, str):
            raise TypeError("Need a width when using a predefined window.")
        if width is not None and not isinstance(window, str):
            raise TypeError("Can only specify a width for a predefined window")

        if isinstance(window, str):
            if window == "gaussian":
                # basically Gaussian theoretically spans infinite time, but practically it falls off quickly,
                # So we choose a window of +- 2*(Standard deviations), i.e 95% of gaussian curve

                width_dt = int(
                    np.round(2 * width / self.clock.dt)
                )  # Rounding only for the size of the window, not for the standard
                # deviation of the Gaussian
                window = np.exp(
                    -np.arange(-width_dt, width_dt + 1) ** 2
                    * 1.0  # hack to ensure floating-point division :)
                    / (2 * (width / self.clock.dt) ** 2)
                )
            elif window == "flat":
                width_dt = int(np.round(width / (2 * self.clock.dt))) * 2 + 1
                used_width = width_dt * self.clock.dt
                if abs(used_width - width) > 1e-6 * self.clock.dt:
                    logger.info(
                        f"width adjusted from {width} to {used_width}",
                        "adjusted_width",
                        once=True,
                    )
                window = np.ones(width_dt)
            else:
                raise NotImplementedError(f'Unknown pre-defined window "{window}"')
        else:
            try:
                window = np.asarray(window)
            except TypeError:
                raise TypeError(f"Cannot use a window of type {type(window)}")
            if window.ndim != 1:
                raise TypeError("The provided window has to be one-dimensional.")
            if len(window) % 2 != 1:
                raise TypeError("The window has to have an odd number of values.")

        # Get the binned rates at the finest resolution
        _, binned_values = self.binned_rate(bin_size=self.clock.dt)

        # Normalize the window
        window = window * 1.0 / sum(window)

        # Extract the raw numpy array from the Quantity (if it is one)
        binned_array = np.asarray(binned_values)

        # So we need to handle for both 1D (PopulationRateMonitor) and 2D (EventMonitor) cases separately as `np.convolve()` only works with 1D arrays
        if binned_values.ndim == 1:
            # PopulationRateMonitor case - 1D convolution
            smoothed = np.convolve(binned_values, window, mode="same")
        else:
            # EventMonitor/SpikeMonitor case - convolve each neuron and then we return the smoothed 2D array ( neuron * bins )
            num_neurons, num_bins = binned_array.shape
            smoothed = np.zeros((num_neurons, num_bins))
            for i in range(num_neurons):
                smoothed[i, :] = np.convolve(binned_array[i, :], window, mode="same")

        return Quantity(smoothed, dim=hertz.dim)


class PopulationRateMonitor(RateMonitor):
    """
    Record instantaneous firing rates, averaged across neurons from a
    `NeuronGroup` or other spike source.

    Parameters
    ----------
    source : (`NeuronGroup`, `SpikeSource`)
        The source of spikes to record.
    name : str, optional
        A unique name for the object, otherwise will use
        ``source.name+'_ratemonitor_0'``, etc.
    codeobj_class : class, optional
        The `CodeObject` class to run code with.
    dtype : dtype, optional
        The dtype to use to store the ``rate`` variable. Defaults to
        `~numpy.float64`, i.e. double precision.

    Notes
    -----
    Currently, this monitor can only monitor the instantaneous firing rates at
    each time step of the source clock. Any binning/smoothing of the firing
    rates has to be done afterwards using the `binned_rate` or `smooth_rate`
    methods.
    """

    invalidates_magic_network = False
    add_to_magic_network = True

    def __init__(
        self, source, name="ratemonitor*", codeobj_class=None, dtype=np.float64
    ):
        #: The group we are recording from
        self.source = source

        self.codeobj_class = codeobj_class
        CodeRunner.__init__(
            self,
            group=self,
            code="",
            template="ratemonitor",
            clock=source.clock,
            when="end",
            order=0,
            name=name,
        )

        self.add_dependency(source)

        self.variables = Variables(self)
        # Handle subgroups correctly
        start = getattr(source, "start", 0)
        stop = getattr(source, "stop", len(source))
        self.variables.add_constant("_source_start", start)
        self.variables.add_constant("_source_stop", stop)
        self.variables.add_reference("_spikespace", source)
        self.variables.add_dynamic_array(
            "rate", size=0, dimensions=hertz.dim, read_only=True, dtype=dtype
        )
        self.variables.add_dynamic_array(
            "t",
            size=0,
            dimensions=second.dim,
            read_only=True,
            dtype=self._clock.variables["t"].dtype,
        )
        self.variables.add_reference("_num_source_neurons", source, "N")
        self.variables.add_array(
            "N", dtype=np.int32, size=1, scalar=True, read_only=True
        )
        self.variables.create_clock_variables(self._clock, prefix="_clock_")
        self._enable_group_attributes()

    def resize(self, new_size):
        # Note that this does not set N, this has to be done in the template
        # since we use a restricted pointer to access it (which promises that
        # we only change the value through this pointer)
        self.variables["rate"].resize(new_size)
        self.variables["t"].resize(new_size)

    def reinit(self):
        """
        Clears all recorded rates
        """
        raise NotImplementedError()

    @check_units(bin_size=second)
    def binned_rate(self, bin_size):
        """
        Return the population rate binned with the given bin size.

        Parameters
        ----------
        bin_size : `Quantity`
            The size of the bins in seconds. Should be a multiple of dt.

        Returns
        -------
        bins : `Quantity`
            The start time of the bins.
        binned_values : `Quantity`
            The binned population rates as a 1D array in Hz.

        Notes
        -----
        The returned bin times represent the **start** of each bin interval, not the center.
        This is consistent with how Brian2 records spike times and other temporal data.
        For example, a spike recorded at time `t` occurred during the interval `[t, t+dt)`.

        For plotting purposes, especially with larger bin sizes, you may want to use bin
        centers instead of bin starts for a more intuitive visualization. You can easily
        calculate the bin centers by adding half the bin size::

            >> bins, rates = monitor.binned_rate(10*ms)
            >> bin_centers = bins + 10*ms / 2
            >> plt.plot(bin_centers, rates)

        This adjustment is particularly helpful when the bins are large relative to the
        time scale of interest, as it better represents where the rate measurement applies
        within each time window.

        Warnings
        --------
        This method will give incorrect results if the monitor has recorded values with
        varying ``dt`` values.
        """
        if (bin_size / self.clock.dt) % 1 > 1e-6:
            raise ValueError("bin_size has to be a multiple of dt.")

        bin_timesteps = timestep(
            bin_size, self.clock.dt
        )  # Convert bin_size to integer timestep

        if (
            bin_timesteps == 1
        ):  # Early return for dt-sized bins (no binning needed) as bin_size and clock timesteps are same
            return self.t[:], self.rate

        # Calculate number of complete bins based on recorded data , Note we don't use `self.clock.timestep` as we want the recorded rates
        num_bins = int(
            len(self.rate) // bin_timesteps
        )  # int for type conversion from numpy int

        num_values = num_bins * bin_timesteps
        rate_to_bin = self.rate[:num_values]  # # Only use complete bins

        # No we reshape into (num_bins, bin_timesteps) and take mean over each bin
        binned_values = rate_to_bin.reshape(num_bins, bin_timesteps).mean(axis=1)

        # Calculate bin centers based on ACTUAL recorded times
        # Start from when recording began, not from t=0
        t_start = self.t[0]
        bin_starts_timesteps = (np.arange(num_bins)) * bin_timesteps
        bins = t_start + bin_starts_timesteps * self.clock.dt

        return bins, Quantity(binned_values, dim=hertz.dim)

    def __repr__(self):
        classname = self.__class__.__name__
        return f"<{classname}, recording {self.source.name}>"
