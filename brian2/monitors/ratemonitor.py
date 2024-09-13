"""
Module defining `PopulationRateMonitor`.
"""

import numpy as np

from brian2.core.names import Nameable
from brian2.core.variables import Variables
from brian2.groups.group import CodeRunner, Group
from brian2.groups.subgroup import Subgroup
from brian2.units.allunits import hertz, second
from brian2.units.fundamentalunits import Quantity, check_units
from brian2.utils.logger import get_logger

__all__ = ["PopulationRateMonitor"]


logger = get_logger(__name__)


class PopulationRateMonitor(Group, CodeRunner):
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
    rates has to be done manually afterwards.
    """

    invalidates_magic_network = False
    add_to_magic_network = True

    def __init__(
        self, source, name="ratemonitor*", codeobj_class=None, dtype=np.float64
    ):
        Nameable.__init__(self, name=name)
        #: The group we are recording from
        self.source = source

        self.codeobj_class = codeobj_class

        self.variables = Variables(self)

        # Handle subgroups correctly
        subgroup = isinstance(source, Subgroup)
        contiguous = not subgroup or source.contiguous
        self.variables.add_arange("_source_idx", size=len(source))
        needed_variables = {}
        if subgroup:
            if contiguous:
                self.variables.add_constant("_source_start", source.start)
                self.variables.add_constant("_source_stop", source.stop)
                needed_variables = {"_source_start", "_source_stop"}
            else:
                self.variables.add_reference(
                    "_source_indices", source, "_sub_idx", index="_source_idx"
                )
                needed_variables = {"_source_indices"}
        self.variables.add_dynamic_array(
            "rate", size=0, dimensions=hertz.dim, read_only=True, dtype=dtype
        )
        CodeRunner.__init__(
            self,
            group=self,
            code="",
            template="ratemonitor",
            clock=source.clock,
            when="end",
            order=0,
            name=name,
            needed_variables=needed_variables,
            template_kwds={
                "subgroup": subgroup,
                "contiguous": contiguous,
                "source_N": source.N,
            },
        )
        self.variables.create_clock_variables(self._clock, prefix="_clock_")
        self.variables.add_dynamic_array(
            "t",
            size=0,
            dimensions=second.dim,
            read_only=True,
            dtype=self._clock.variables["t"].dtype,
        )
        self.variables.add_array(
            "N", dtype=np.int32, size=1, scalar=True, read_only=True
        )
        self.variables.add_reference("_spikespace", source)

        self.add_dependency(source)
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

    @check_units(width=second)
    def smooth_rate(self, window="gaussian", width=None):
        """
        smooth_rate(self, window='gaussian', width=None)

        Return a smooth version of the population rate.

        Parameters
        ----------
        window : str, ndarray
            The window to use for smoothing. Can be a string to chose a
            predefined window(``'flat'`` for a rectangular, and ``'gaussian'``
            for a Gaussian-shaped window). In this case the width of the window
            is determined by the ``width`` argument. Note that for the Gaussian
            window, the ``width`` parameter specifies the standard deviation of
            the Gaussian, the width of the actual window is ``4*width + dt``
            (rounded to the nearest dt). For the flat window, the width is
            rounded to the nearest odd multiple of dt to avoid shifting the rate
            in time.
            Alternatively, an arbitrary window can be given as a numpy array
            (with an odd number of elements). In this case, the width in units
            of time depends on the ``dt`` of the simulation, and no ``width``
            argument can be specified. The given window will be automatically
            normalized to a sum of 1.
        width : `Quantity`, optional
            The width of the ``window`` in seconds (for a predefined window).

        Returns
        -------
        rate : `Quantity`
            The population rate in Hz, smoothed with the given window. Note that
            the rates are smoothed and not re-binned, i.e. the length of the
            returned array is the same as the length of the ``rate`` attribute
            and can be plotted against the `PopulationRateMonitor` 's ``t``
            attribute.
        """
        if width is None and isinstance(window, str):
            raise TypeError("Need a width when using a predefined window.")
        if width is not None and not isinstance(window, str):
            raise TypeError("Can only specify a width for a predefined window")

        if isinstance(window, str):
            if window == "gaussian":
                width_dt = int(np.round(2 * width / self.clock.dt))
                # Rounding only for the size of the window, not for the standard
                # deviation of the Gaussian
                window = np.exp(
                    -np.arange(-width_dt, width_dt + 1) ** 2
                    * 1.0
                    / (2 * (width / self.clock.dt) ** 2)
                )
            elif window == "flat":
                width_dt = int(width / 2 / self.clock.dt) * 2 + 1
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
        return Quantity(
            np.convolve(self.rate_, window * 1.0 / sum(window), mode="same"),
            dim=hertz.dim,
        )

    def __repr__(self):
        classname = self.__class__.__name__
        return f"<{classname}, recording {self.source.name}>"
