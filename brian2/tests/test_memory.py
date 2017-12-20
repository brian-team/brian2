from nose.plugins.attrib import attr
import numpy as np
from numpy.testing.utils import assert_equal

from brian2.memory.dynamicarray import DynamicArray, DynamicArray1D

@attr('codegen-independent')
def test_dynamic_array_1d_access():
    da = DynamicArray1D(10)
    da[:] = np.arange(10)
    assert da[7] == 7
    assert len(da) == 10
    assert da.shape == (10, )
    assert len(str(da))
    assert len(repr(da))
    da[:] += 1
    da.data[:] += 1
    assert all(da[:] == (np.arange(10) + 2))


@attr('codegen-independent')
def test_dynamic_array_1d_resize_up_down():
    for numpy_resize in [True, False]:
        da = DynamicArray1D(10, use_numpy_resize=numpy_resize)
        da[:] = np.arange(10)
        da.resize(15)
        assert len(da) == 15
        assert da.shape == (15, )
        assert all(da[10:] == 0)
        assert all(da[:10] == np.arange(10))
        da.resize(5)
        assert len(da) == 5
        assert da.shape == (5, )
        assert all(da[:] == np.arange(5))

@attr('codegen-independent')
def test_dynamic_array_1d_resize_down_up():
    for numpy_resize in [True, False]:
        da = DynamicArray1D(10, use_numpy_resize=numpy_resize)
        da[:] = np.arange(10)
        da.resize(5)
        assert len(da) == 5
        assert da.shape == (5, )
        assert all(da[:5] == np.arange(5))
        da.resize(10)
        assert len(da) == 10
        assert da.shape == (10, )
        assert all(da[:5] == np.arange(5))
        assert all(da[5:] == 0)

@attr('codegen-independent')
def test_dynamic_array_1d_shrink():
    for numpy_resize in [True, False]:
        da = DynamicArray1D(10, use_numpy_resize=numpy_resize)
        da[:] = np.arange(10)
        da.shrink(5)
        assert len(da) == 5
        assert all(da[:] == np.arange(5))
        # After using shrink, the underlying array should have changed
        assert len(da._data) == 5


@attr('codegen-independent')
def test_dynamic_array_2d_access():
    da = DynamicArray1D((10, 20))
    da[:, :] = np.arange(200).reshape((10, 20))
    assert da[5, 10] == 5*20 + 10
    assert da.shape == (10, 20)
    assert len(str(da))
    assert len(repr(da))
    da[:] += 1
    da.data[:] += 1
    assert_equal(da[:, :], np.arange(200).reshape((10, 20)) + 2)


@attr('codegen-independent')
def test_dynamic_array_2d_resize_up_down():
    for numpy_resize in [True, False]:
        da = DynamicArray((10, 20), use_numpy_resize=numpy_resize)
        da[:, :] = np.arange(200).reshape((10, 20))
        da.resize((15, 20))
        assert da.shape == (15, 20)
        assert_equal(da[10:, :], np.zeros((5, 20)))
        assert_equal(da[:10, :], np.arange(200).reshape((10, 20)))
        da.resize((15, 25))
        assert da.shape == (15, 25)
        assert_equal(da[:10, 20:], np.zeros((10, 5)))
        assert_equal(da[:10, :20], np.arange(200).reshape((10, 20)))

        da.resize((10, 20))
        assert da.shape == (10, 20)
        assert_equal(da[:, :], np.arange(200).reshape((10, 20)))


@attr('codegen-independent')
def test_dynamic_array_2d_resize_down_up():
    for numpy_resize in [True, False]:
        da = DynamicArray((10, 20), use_numpy_resize=numpy_resize)
        da[:, :] = np.arange(200).reshape((10, 20))
        da.resize((5, 20))
        assert da.shape == (5, 20)
        assert_equal(da, np.arange(100).reshape((5, 20)))
        da.resize((5, 15))
        assert da.shape == (5, 15)
        for row_idx, row in enumerate(da):
            assert_equal(row, 20*row_idx + np.arange(15))

        da.resize((10, 20))
        assert da.shape == (10, 20)
        for row_idx, row in enumerate(da[:5, :15]):
            assert_equal(row, 20*row_idx + np.arange(15))
        assert_equal(da[5:, 15:], 0)


@attr('codegen-independent')
def test_dynamic_array_2d_shrink():
    for numpy_resize in [True, False]:
        da = DynamicArray((10, 20), use_numpy_resize=numpy_resize)
        da[:, :] = np.arange(200).reshape((10, 20))
        da.shrink((5, 15))
        assert da.shape == (5, 15)
        # After using shrink, the underlying array should have changed
        assert da._data.shape == (5, 15)
        assert_equal(da[:, :], np.arange(15).reshape((1, 15)) + 20*np.arange(5).reshape((5, 1)))


if __name__=='__main__':
    test_dynamic_array_1d_access()
    test_dynamic_array_1d_resize_up_down()
    test_dynamic_array_1d_resize_down_up()
    test_dynamic_array_1d_shrink()
    test_dynamic_array_2d_access()
    test_dynamic_array_2d_resize_up_down()
    test_dynamic_array_2d_resize_down_up()
    test_dynamic_array_2d_shrink()
