#ifndef _BRIAN_CDYNAMICARRAY_H
#define _BRIAN_CDYNAMICARRAY_H

#include <vector>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <cassert>
#include <memory>

template<typename T>
class CDynamicArray {
private:
    std::vector<T> _data;
    std::vector<size_t> _shape;
    std::vector<size_t> _strides;
    size_t _allocated_size;
    size_t _ndim;
    double _factor;

    void compute_strides() {
        if (_ndim == 0) return;

        _strides.resize(_ndim);
        _strides[_ndim - 1] = 1;
        for (int i = _ndim - 2; i >= 0; --i) {
            _strides[i] = _strides[i + 1] * _shape[i + 1];
        }
    }

    size_t compute_total_size() const {
        size_t size = 1;
        for (size_t dim : _shape) {
            size *= dim;
        }
        return size;
    }

    size_t compute_allocated_size() const {
        size_t size = 1;
        for (size_t i = 0; i < _ndim; ++i) {
            size *= (i < _shape.size() ? _shape[i] : 1);
        }
        return size;
    }

public:
    CDynamicArray(const std::vector<size_t>& shape, double factor = 2.0)
        : _shape(shape), _ndim(shape.size()), _factor(factor) {

        _allocated_size = compute_total_size();
        _data.resize(_allocated_size);
        compute_strides();

        // Initialize with zeros
        std::fill(_data.begin(), _data.end(), T(0));
    }

    // Constructor for 1D array
    CDynamicArray(size_t size, double factor = 2.0)
        : _shape({size}), _ndim(1), _factor(factor) {

        _allocated_size = size;
        _data.resize(_allocated_size);
        _strides = {1};

        // Initialize with zeros
        std::fill(_data.begin(), _data.end(), T(0));
    }

    ~CDynamicArray() = default;

    // Get raw data pointer
    T* data() { return _data.data(); }
    const T* data() const { return _data.data(); }

    const std::vector<size_t>& shape() const { return _shape; }
    const std::vector<size_t>& strides() const { return _strides; }
    size_t ndim() const { return _ndim; }

    size_t size() const { return compute_total_size(); }

    // Resize the array
    void resize(const std::vector<size_t>& new_shape) {
        assert(new_shape.size() == _ndim);

        size_t new_size = 1;
        for (size_t dim : new_shape) {
            new_size *= dim;
        }

        // Check if we need to allocate more memory
        if (new_size > _allocated_size) {
            // Calculate new allocated size with growth factor
            size_t target_size = static_cast<size_t>(_allocated_size * _factor);
            _allocated_size = std::max(target_size, new_size);

            // Create new data array
            std::vector<T> new_data(_allocated_size, T(0));

            // Copy old data (handling multi-dimensional copy)
            if (_ndim == 1) {
                // Simple 1D copy
                size_t copy_size = std::min(_shape[0], new_shape[0]);
                std::copy(_data.begin(), _data.begin() + copy_size, new_data.begin());
            } else {
                // Multi-dimensional copy - need to handle stride changes
                copy_data_multidim(_data.data(), new_data.data(), _shape, new_shape, _strides);
            }

            _data = std::move(new_data);
        } else if (new_size < compute_total_size()) {
            // Clear the unused portion
            size_t old_size = compute_total_size();
            std::fill(_data.begin() + new_size, _data.begin() + old_size, T(0));
        }

        _shape = new_shape;
        compute_strides();
    }

    void resize_1d(size_t new_size) {
        assert(_ndim == 1);

        if (new_size > _allocated_size) {
            size_t target_size = static_cast<size_t>(_allocated_size * _factor);
            _allocated_size = std::max(target_size, new_size);
            _data.resize(_allocated_size, T(0));
        } else if (new_size < _shape[0]) {
            std::fill(_data.begin() + new_size, _data.begin() + _shape[0], T(0));
        }

        _shape[0] = new_size;
    }

    // Shrink to exact size (deallocates extra memory)
    void shrink(const std::vector<size_t>& new_shape) {
        assert(new_shape.size() == _ndim);

        size_t new_size = 1;
        for (size_t dim : new_shape) {
            new_size *= dim;
        }

        std::vector<T> new_data(new_size);

        if (_ndim == 1) {
            size_t copy_size = std::min(_shape[0], new_shape[0]);
            std::copy(_data.begin(), _data.begin() + copy_size, new_data.begin());
        } else {
            copy_data_multidim(_data.data(), new_data.data(), _shape, new_shape, _strides);
        }

        _data = std::move(new_data);
        _shape = new_shape;
        _allocated_size = new_size;
        compute_strides();
    }

    // Access element at given indices
    T& operator()(const std::vector<size_t>& indices) {
        assert(indices.size() == _ndim);
        size_t offset = 0;
        for (size_t i = 0; i < _ndim; ++i) {
            assert(indices[i] < _shape[i]);
            offset += indices[i] * _strides[i];
        }
        return _data[offset];
    }

    // 1D access
    T& operator[](size_t index) {
        assert(_ndim == 1 && index < _shape[0]);
        return _data[index];
    }

    // Get/set slices (for Python interface)
    void get_slice(T* output, const std::vector<std::pair<int, int>>& slices) const {
        // Implementation for extracting slices
        // TODO: Will make this better
        if (_ndim == 1 && slices.size() == 1) {
            int start = slices[0].first;
            int stop = slices[0].second;
            if (start < 0) start = 0;
            if (stop > static_cast<int>(_shape[0])) stop = _shape[0];
            std::copy(_data.begin() + start, _data.begin() + stop, output);
        }
        // TODO: Add more complex slicing logic as needed
    }

    void set_slice(const T* input, const std::vector<std::pair<int, int>>& slices) {
        // Implementation for setting slices
        if (_ndim == 1 && slices.size() == 1) {
            int start = slices[0].first;
            int stop = slices[0].second;
            if (start < 0) start = 0;
            if (stop > static_cast<int>(_shape[0])) stop = _shape[0];
            std::copy(input, input + (stop - start), _data.begin() + start);
        }
        //TODO: Add more complex slicing logic as needed
    }

private:
    // Helper function for multi-dimensional copy
    void copy_data_multidim(const T* src, T* dst,
                           const std::vector<size_t>& src_shape,
                           const std::vector<size_t>& dst_shape,
                           const std::vector<size_t>& src_strides) {
        // TODO: proper implementation should handle
        // all cases of multi-dimensional copying with different strides
        std::vector<size_t> min_shape(src_shape.size());
        for (size_t i = 0; i < src_shape.size(); ++i) {
            min_shape[i] = std::min(src_shape[i], dst_shape[i]);
        }

        // For 2D case as example
        if (_ndim == 2) {
            for (size_t i = 0; i < min_shape[0]; ++i) {
                for (size_t j = 0; j < min_shape[1]; ++j) {
                    dst[i * dst_shape[1] + j] = src[i * src_strides[0] + j];
                }
            }
        }
        // TODO: generalize for arbitrary dimensions
    }
};


template<typename T>
class CDynamicArray1D {
private:
    std::vector<T> _data;
    size_t _size;
    size_t _allocated_size;
    double _factor;

public:
    CDynamicArray1D(size_t size, double factor = 2.0)
        : _size(size), _allocated_size(size), _factor(factor) {
        _data.resize(_allocated_size, T(0));
    }

    T* data() { return _data.data(); }
    const T* data() const { return _data.data(); }

    size_t size() const { return _size; }

    void resize(size_t new_size) {
        if (new_size > _allocated_size) {
            size_t target_size = static_cast<size_t>(_allocated_size * _factor);
            _allocated_size = std::max(target_size, new_size);
            _data.resize(_allocated_size, T(0));
        } else if (new_size < _size) {
            std::fill(_data.begin() + new_size, _data.begin() + _size, T(0));
        }
        _size = new_size;
    }

    void shrink(size_t new_size) {
        assert(new_size <= _size);
        std::vector<T> new_data(new_size);
        std::copy(_data.begin(), _data.begin() + new_size, new_data.begin());
        _data = std::move(new_data);
        _size = new_size;
        _allocated_size = new_size;
    }

    T& operator[](size_t index) {
        assert(index < _size);
        return _data[index];
    }

    const T& operator[](size_t index) const {
        assert(index < _size);
        return _data[index];
    }
};

#endif // _BRIAN_CDYNAMICARRAY_H
#
