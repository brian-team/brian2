'''
Implementation of `PoissonInput`.
'''
from .binomial import BinomialFunction

from brian2.core.variables import Variables
from brian2.groups.group import CodeRunner
from brian2.units.fundamentalunits import (check_units, have_same_dimensions,
                                           get_unit, Quantity,
                                           DimensionMismatchError)
from brian2.units.stdunits import Hz


__all__ = ['PoissonInput']


class PoissonInput(CodeRunner):
    '''
    PoissonInput(target, target_var, N, rate, weight, when='synapses', order=0)

    Adds independent Poisson input to a target variable of a `Group`. For large
    numbers of inputs, this is much more efficient than creating a
    `PoissonGroup`. The synaptic events are generated randomly during the
    simulation and are not preloaded and stored in memory. All the inputs must
    target the same variable, have the same frequency and same synaptic weight.
    All neurons in the target `Group` receive independent realizations of
    Poisson spike trains.

    Parameters
    ----------
    target : `Group`
        The group that is targeted by this input.
    target_var : str
        The variable of `target` that is targeted by this input.
    N : int
        The number of inputs
    rate : `Quantity`
        The rate of each of the inputs
    weight : str or `Quantity`
        Either a string expression (that can be interpreted in the context of
        `target`) or a `Quantity` that will be added for every event to
        the `target_var` of `target`. The unit has to match the unit of
        `target_var`
    when : str, optional
        When to update the target variable during a time step. Defaults to
        the `synapses` scheduling slot.
    order : int, optional
        The priority of of the update compared to other operations occurring at
        the same time step and in the same scheduling slot. Defaults to 0.

    '''
    @check_units(N=1, rate=Hz)
    def __init__(self, target, target_var, N, rate, weight, when='synapses',
                 order=0):
        if target_var not in target.variables:
            raise KeyError('%s is not a variable of %s' % (target_var, target.name))

        if isinstance(weight, basestring):
            weight = '(%s)' % weight
        else:
            weight_unit = get_unit(weight)
            weight = repr(weight)
            target_unit = target.variables[target_var].unit
            # This will be checked automatically in the abstract code as well
            # but doing an explicit check here allows for a clearer error
            # message
            if not have_same_dimensions(weight_unit, target_unit):
                raise DimensionMismatchError(('The provided weight does not '
                                              'have the same unit as the '
                                              'target variable "%s"') % target_var,
                                             weight_unit.dim,
                                             target_unit.dim)


        binomial_sampling = BinomialFunction(N, rate*target.clock.dt,
                                             name='poissoninput_binomial*')

        code = '{targetvar} += {binomial}()*{weight}'.format(targetvar=target_var,
                                                             binomial=binomial_sampling.name,
                                                             weight=weight)
        self._stored_dt = target.dt_[:]  # make a copy
        # FIXME: we need an explicit reference here for on-the-fly subgroups
        # For example: PoissonInput(group[:N], ...)
        self._group = target
        CodeRunner.__init__(self,
                            group=target,
                            template='stateupdate',
                            code=code,
                            user_code='',
                            when=when,
                            order=order,
                            name='poissoninput*',
                            clock=target.clock
                            )
        self.variables = Variables(self)
        self.variables._add_variable(binomial_sampling.name, binomial_sampling)

    def before_run(self, run_namespace):
        if self._group.dt_ != self._stored_dt:
            raise NotImplementedError('The dt used for simulating %s changed '
                                      'after the PoissonInput source was '
                                      'created.' % self.group.name)
        CodeRunner.before_run(self, run_namespace=run_namespace)
