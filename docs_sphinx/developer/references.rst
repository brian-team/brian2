References to Brian objects
===========================

weak references (weak proxies)

    For references that would lead to a circular reference. This should only
    apply to objects that are not directly accessed by the user, e.g.
    `CodeObject` or `Variable` objects.

`Proxy` objects

    For references to objects derived from `BrianObject`. Should be used when
    the reference to the object might still be used after the object disappeared
    but by itself should not make the object simulated as part of
    `MagicNetwork`. Examples for this use case are the references to a
    `Group` stored in a monitor or in `VariableView`. A function might return
    a reference to a state variable of a `Group` (i.e., a `VariableView`) and
    this should remain useable even when there is no more explicit reference to
    the `Group`. On the other hand, `MagicNetwork` should not continue to
    simulate the `Group`.

strong (standard) references

    This is used in `Network` for explicitly added objects. This allows it to
    add an object to a network without keeping a reference to it, e.g. to do
    ``net.add(SpikeMonitor(group))``.
