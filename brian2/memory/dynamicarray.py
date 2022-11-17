"""
TODO: rewrite this (verbatim from Brian 1.x), more efficiency
"""

from numpy import *

__all__ = ["DynamicArray", "DynamicArray1D"]


def getslices(shape, from_start=True):
    if from_start:
        return tuple(slice(0, x) for x in shape)
    else:
        return tuple(slice(x, None) for x in shape)


class DynamicArray(object):
    """
    An N-dimensional dynamic array class

    The array can be resized in any dimension, and the class will handle
    allocating a new block of data and copying when necessary.

    .. warning::
        The data will NOT be contiguous for >1D arrays. To ensure this, you will
        either need to use 1D arrays, or to copy the data, or use the shrink
        method with the current size (although note that in both cases you
        negate the memory and efficiency benefits of the dynamic array).

    Initialisation arguments:

    ``shape``, ``dtype``
        The shape and dtype of the array to initialise, as in Numpy. For 1D
        arrays, shape can be a single int, for ND arrays it should be a tuple.
    ``factor``
        The resizing factor (see notes below). Larger values tend to lead to
        more wasted memory, but more computationally efficient code.
    ``use_numpy_resize``, ``refcheck``
        Normally, when you resize the array it creates a new array and copies
        the data. Sometimes, it is possible to resize an array without a copy,
        and if this option is set it will attempt to do this. However, this can
        cause memory problems if you are not careful so the option is off by
        default. You need to ensure that you do not create slices of the array
        so that no references to the memory exist other than the main array
        object. If you are sure you know what you're doing, you can switch this
        reference check off. Note that resizing in this way is only done if you
        resize in the first dimension.

    The array is initialised with zeros. The data is stored in the attribute
    ``data`` which is a Numpy array.


    Some numpy methods are implemented and can work directly on the array object,
    including ``len(arr)``, ``arr[...]`` and ``arr[...]=...``. In other cases,
    use the ``data`` attribute.

    Examples
    --------

    >>> x = DynamicArray((2, 3), dtype=int)
    >>> x[:] = 1
    >>> x.resize((3, 3))
    >>> x[:] += 1
    >>> x.resize((3, 4))
    >>> x[:] += 1
    >>> x.resize((4, 4))
    >>> x[:] += 1
    >>> x.data[:] = x.data**2
    >>> x.data
    array([[16, 16, 16,  4],
           [16, 16, 16,  4],
           [ 9,  9,  9,  4],
           [ 1,  1,  1,  1]])

    Notes
    -----

    The dynamic array returns a ``data`` attribute which is a view on the larger
    ``_data`` attribute. When a resize operation is performed, and a specific
    dimension is enlarged beyond the size in the ``_data`` attribute, the size
    is increased to the larger of ``cursize*factor`` and ``newsize``. This
    ensures that the amortized cost of increasing the size of the array is O(1).
    """

    def __init__(
        self, shape, dtype=float, factor=2, use_numpy_resize=False, refcheck=True
    ):
        if isinstance(shape, int):
            shape = (shape,)
        self._data = zeros(shape, dtype=dtype)
        self.data = self._data
        self.dtype = dtype
        self.shape = self._data.shape
        self.factor = factor
        self.use_numpy_resize = use_numpy_resize
        self.refcheck = refcheck

    def resize(self, newshape):
        """
        Resizes the data to the new shape, which can be a different size to the
        current data, but should have the same rank, i.e. same number of
        dimensions.
        """
        datashapearr = array(self._data.shape)
        newshapearr = array(newshape)
        resizedimensions = newshapearr > datashapearr
        if resizedimensions.any():
            # resize of the data is needed
            minnewshapearr = datashapearr  # .copy()
            dimstoinc = minnewshapearr[resizedimensions]
            incdims = array(dimstoinc * self.factor, dtype=int)
            newdims = maximum(incdims, dimstoinc + 1)
            minnewshapearr[resizedimensions] = newdims
            newshapearr = maximum(newshapearr, minnewshapearr)
            do_resize = False
            if self.use_numpy_resize and self._data.flags["C_CONTIGUOUS"]:
                if sum(resizedimensions) == resizedimensions[0]:
                    do_resize = True
            if do_resize:
                self.data = None
                self._data.resize(tuple(newshapearr), refcheck=self.refcheck)
            else:
                newdata = zeros(tuple(newshapearr), dtype=self.dtype)
                slices = getslices(self._data.shape)
                newdata[slices] = self._data
                self._data = newdata
        elif (newshapearr < self.shape).any():
            # If we reduced the size, set the no longer used memory to 0
            self._data[getslices(newshape, from_start=False)] = 0
        # Reduce our view to the requested size if necessary
        self.data = self._data[getslices(newshape, from_start=True)]
        self.shape = self.data.shape

    def resize_along_first(self, newshape):
        new_dimension = newshape[0]
        if new_dimension > self._data.shape[0]:
            new_size = maximum(self._data.shape[0] * self.factor, new_dimension + 1)
            final_new_shape = array(self._data.shape)
            final_new_shape[0] = new_size
            if self.use_numpy_resize and self._data.flags["C_CONTIGUOUS"]:
                self.data = None
                self._data.resize(tuple(final_new_shape), refcheck=self.refcheck)
            else:
                newdata = zeros(tuple(final_new_shape), dtype=self.dtype)
                slices = getslices(self._data.shape)
                newdata[slices] = self._data
                self._data = newdata
        elif newshape < self.shape:
            # If we reduced the size, set the no longer used memory to 0
            self._data[new_dimension:] = 0
        # Reduce our view to the requested size if necessary
        self.data = self._data[:new_dimension]
        self.shape = newshape

    def shrink(self, newshape):
        """
        Reduces the data to the given shape, which should be smaller than the
        current shape. `resize` can also be used with smaller values, but
        it will not shrink the allocated memory, whereas `shrink` will
        reallocate the memory. This method should only be used infrequently, as
        if it is used frequently it will negate the computational efficiency
        benefits of the DynamicArray.
        """
        if isinstance(newshape, int):
            newshape = (newshape,)
        shapearr = array(self.shape)
        newshapearr = array(newshape)
        if (newshapearr <= shapearr).all():
            newdata = zeros(newshapearr, dtype=self.dtype)
            newdata[:] = self._data[getslices(newshapearr)]
            self._data = newdata
            self.shape = tuple(newshapearr)
            self.data = self._data

    def __getitem__(self, item):
        return self.data.__getitem__(item)

    def __setitem__(self, item, val):
        self.data.__setitem__(item, val)

    def __len__(self):
        return len(self.data)

    def __str__(self):
        return self.data.__str__()

    def __repr__(self):
        return self.data.__repr__()


class DynamicArray1D(DynamicArray):
    """
    Version of `DynamicArray` with specialised ``resize`` method designed
    to be more efficient.
    """

    def resize(self, newshape):
        (datashape,) = self._data.shape
        if newshape > datashape:
            (shape,) = self.shape  # we work with int shapes only
            newdatashape = max(newshape, int(shape * self.factor) + 1)
            if self.use_numpy_resize and self._data.flags["C_CONTIGUOUS"]:
                self.data = None
                self._data.resize(newdatashape, refcheck=self.refcheck)
            else:
                newdata = zeros(newdatashape, dtype=self.dtype)
                newdata[:shape] = self.data
                self._data = newdata
        elif newshape < self.shape[0]:
            # If we reduced the size, set the no longer used memory to 0
            self._data[newshape:] = 0
        # Reduce our view to the requested size if necessary
        self.data = self._data[:newshape]
        self.shape = (newshape,)
