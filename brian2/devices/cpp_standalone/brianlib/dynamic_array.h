#ifndef _BRIAN_DYNAMIC_ARRAY_H
#define _BRIAN_DYNAMIC_ARRAY_H

#include <cstddef>
#include <type_traits>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cassert>

/**
 * A simple 1D dynamic array that grows efficiently over time.
 *
 * This class is designed to mimic the behavior of C-style contiguous memory,
 * making it suitable for interop with tools like Cython and NumPy.
 *
 * Internally, it keeps track of:
 * - `m_size`: the number of elements the user is actively working with.
 * - `m_data.capacity()`: the total number of elements currently allocated.
 *
 * When growing, it over-allocates using a growth factor to avoid frequent
 * memory reallocations — giving us amortized O(1) behavior for appending elements.
 *
 * When shrinking, it simply zeroes out the unused portion instead of
 * releasing memory immediately. To actually free that extra memory,
 * call `shrink_to_fit()`.
 */
template <class T>
class DynamicArray1D
{
private:
    std::vector<T> m_data;
    size_t m_size; // Logical size (what user sees)
    double m_growth_factor;

public:
    /**
     *
     * We call m_data.resize(initial_size) to ensure that operator[] is safe up to
     * initial_size-1 immediately after construction. This also sets capacity() to
     * at least initial_size.
     */
    DynamicArray1D(size_t initial_size = 0, double factor = 2.0)
        : m_size(initial_size), m_growth_factor(factor)
    {
        m_data.resize(initial_size);
    }

    ~DynamicArray1D(){}; // note earlier we needed a destructor properly because we had a vector of pointers ...

    /**
     * @brief Resizes the array to a new logical size.
     *
     * If the new size is larger than the current capacity, we grow the buffer.
     * To avoid frequent reallocations, we over-allocate using a growth factor—
     * that means the actual buffer might grow more than you asked for.
     * This helps keep future resizes fast (amortized O(1) behavior).
     *
     * If the new size is smaller than the current logical size, we don't shrink
     * the buffer immediately. Instead, we zero out the unused part to avoid
     * keeping stale data around. If you really want to release unused memory,
     * call `shrink_to_fit()` separately.
     */
    void resize(size_t new_size)
    {
        if (new_size > m_data.size())
        {
            // Growing: allocate more than strictly needed to reduce future allocations
            size_t grown = static_cast<size_t>(m_data.size() * m_growth_factor) + 1;
            size_t new_capacity = std::max(new_size, grown);
            m_data.resize(new_capacity);
        }
        else if (new_size < m_size)
        {
            // Shrinking: zero out "deleted" entries for safety
            std::fill(m_data.begin() + new_size,
                      m_data.begin() + m_size,
                      T(0));
        }
        m_size = new_size;
    }

    /**
     * Shrink capacity to match current size
     * Use with precaution as it defeats the purpose of amortized growth
     */
    void shrink_to_fit()
    {
        m_data.resize(m_size);
        m_data.shrink_to_fit();
    }
    size_t size() const { return m_size; }
    size_t capacity() const  { return m_data.size(); }

    /**
     * @brief Direct access to the underlying data pointer.
     * @return Pointer to the first element (may be null if capacity()==0).
     *
     * This be used by us for using the dynamic array with numpy
     */
    T *get_data_ptr()  { return m_data.data(); }
    const T *get_data_ptr() const  { return m_data.data(); }

    T &operator[](size_t idx) { return m_data[idx]; }
    const T &operator[](size_t idx) const  { return m_data[idx]; }
};

/**
 * @brief A two-dimensional dynamic array backed by a flat, row-major buffer.
 *
 * Stores data in a single contiguous std::vector<T> to match C-style and NumPy
 * memory layout, enabling zero-copy interop (e.g., via Cython).
 * Supports amortized , O(1) growth in both dimensions and efficient shrinking.
 */
template <class T>
class DynamicArray2D
{
private:
    std::vector<T> m_buffer; // Underlying flat buffer (capacity = allocated slots)
    size_t m_rows;           // Logical number of rows exposed to the user
    size_t m_cols;           // Logical number of columns exposed to the user
    size_t m_buffer_rows;    // Physical buffer row capacity
    size_t m_buffer_cols;    // Physical buffer column capacity (stride)
    double m_growth_factor;  // Grow multiplier to reduce realloc frequency

    /**
     * Convert 2D coordinates to flat index
     * Row-major: i.e. elements of same row are contiguous
     */
    inline size_t index(size_t i, size_t j) const
    {
        assert(i < m_buffer_rows && j < m_buffer_cols);
        return i * m_buffer_cols + j;
    }

public:
    // We keep these for backwards compatibility
    size_t *n;
    size_t *m;

    DynamicArray2D(size_t rows = 0, size_t cols = 0, double factor = 2.0)
        : m_rows(rows), m_cols(cols),
          m_buffer_rows(rows), m_buffer_cols(cols),
          m_growth_factor(factor),
          n(&m_rows), m(&m_cols)
    {
        m_buffer.resize(m_buffer_rows * m_buffer_cols);
    }
    /**
     * @brief Legacy constructor
     */
    DynamicArray2D(int _n, int _m)
        : DynamicArray2D(static_cast<size_t>(_n),
                         static_cast<size_t>(_m),
                         2.0) {}

    ~DynamicArray2D(){};

    /**
     * @brief Resize the array to new_rows x new_cols, preserving as much data as possible.
     * @param new_rows The desired number of logical rows.
     * @param new_cols The desired number of logical columns.
     *
     * If the requested size is larger than the current buffer, we grow the
     * internal storage using an over-allocation strategy:
     *    new_dim = max(requested, old_capacity * growth_factor + 1)
     * for each dimension. This reduces the number of reallocations over time
     * and provides amortized O(1) growth.
     *
     * When resizing down (shrinking), we *don’t* free memory immediately.
     * Instead, we simply zero out the parts of the buffer that are now
     * outside the logical size. To actually release unused memory,
     * call `shrink_to_fit()`.
     */
    void resize(size_t new_rows, size_t new_cols)
    {
        bool needs_realloc = false;
        size_t grow_rows = m_buffer_rows;
        size_t grow_cols = m_buffer_cols;

        // First we check if buffer needs to grows
        if (new_rows > m_buffer_rows)
        {
            size_t candidate = static_cast<size_t>(m_buffer_rows * m_growth_factor) + 1;
            grow_rows = std::max(new_rows, candidate);
            needs_realloc = true;
        }
        if (new_cols > m_buffer_cols)
        {
            size_t candidate = static_cast<size_t>(m_buffer_cols * m_growth_factor) + 1;
            grow_cols = std::max(new_cols, candidate);
            needs_realloc = true;
        }

        if (needs_realloc)
        {
            // Allocate new buffer and copy existing data
            std::vector<T> new_buf(grow_rows * grow_cols);
            size_t copy_rows = std::min(m_rows, new_rows);
            size_t copy_cols = std::min(m_cols, new_cols);

            for (size_t i = 0; i < copy_rows; ++i)
            {
                for (size_t j = 0; j < copy_cols; ++j)
                {
                    new_buf[i * grow_cols + j] = m_buffer[index(i, j)];
                }
            }
            // Swap in the new buffer and update capacities
            m_buffer.swap(new_buf);
            m_buffer_rows = grow_rows;
            m_buffer_cols = grow_cols;
        }
        else if (new_rows < m_rows || new_cols < m_cols)
        {
            // Efficiently clear only the unused region without reallocating
            // Zero rows beyond new_rows
            for (size_t i = new_rows; i < m_buffer_rows; ++i)
            {
                size_t base = i * m_buffer_cols;
                std::fill(&m_buffer[base], &m_buffer[base + m_buffer_cols], T(0));
            }
            // Zero columns beyond new_cols in remaining rows
            for (size_t i = 0; i < new_rows; ++i)
            {
                size_t base = i * m_buffer_cols + new_cols;
                std::fill(&m_buffer[base], &m_buffer[base + (m_buffer_cols - new_cols)], T(0));
            }
        }

        // Finally, we update logical dimensions
        m_rows = new_rows;
        m_cols = new_cols;
    }

    // Legacy overloads for compatibility
    void resize(int new_n, int new_m)
    {
        resize(static_cast<size_t>(new_n), static_cast<size_t>(new_m));
    }

    void resize()
    {
        resize(m_rows, m_cols);
    }

    /**
    * @brief Efficiently resize only the first dimension (rows) while keeping columns unchanged.
    *
    * @note This method assumes columns remain constant. If you need to change both
    *       dimensions, use the general resize(rows, cols) method instead.
    */
    void resize_along_first(size_t new_rows)
    {
        if(new_rows > m_buffer_rows) // growth case
        {
            //  So first we calculate how much to grow the buffer and then we over-allocate to avoid frequent reallocations
            size_t candidate = static_cast<size_t>(m_buffer_rows * m_growth_factor) + 1;
            size_t grow_rows = std::max(new_rows,candidate);

            // now we create a new buffer with new row capacity , while the column capacity remains same
            std::vector<T> new_buf(grow_rows * m_buffer_cols);

            // Figure out how many rows of existing data we can preserve
            size_t copy_rows = std::min(m_rows, new_rows);

            if ( std::is_trivially_copyable<T>::value && copy_rows > 0)
            {
                // We copy one complete row in a single memcpy operation ... much faster than copying element by element
                for (size_t i = 0; i < copy_rows; ++i)
                {
                    std::memcpy(&new_buf[i*m_buffer_cols], // destination: row i in new buffer
                        &m_buffer[i*m_buffer_cols], // source: row i in old buffer
                        m_buffer_cols * sizeof(T) // size: entire row
                    );
                }
            }
            else
            {
                for (size_t i =0; i< copy_rows; i++)
                {
                    for (size_t j =0; j < m_buffer_cols; ++j) // ++j does not create a copy — it just increments and returns the reference , for iterators and classes, ++j can be significantly faster.
                    {
                        new_buf[i*m_buffer_cols +j] = m_buffer[index(i,j)];
                    }
                }
            }

            m_buffer.swap(new_buf);
            m_buffer_rows = grow_rows;
        }
        else if (new_rows < m_rows) // shrinkage case
        {
            // As we are reducing the number of rows , so we zero out deleted rows
            for ( size_t i = new_rows; i < m_rows ; ++i)
            {
                size_t base = i * m_buffer_cols;

                // Zero out the entire row in one operation
                std::fill(&m_buffer[base], &m_buffer[base + m_buffer_cols],T(0));
            }

           /* Note: We don't shrink the actual buffer capacity here
            * This is intentional for performance - if you're shrinking temporarily,
            * you don't want to pay the cost of reallocation when you grow again.
            * Call shrink_to_fit() explicitly if you need to reclaim memory.
            */
        }
        // We just update the logical row count to reflect the new size
        m_rows =new_rows;
    }

    /**
     * Shrink buffer to exact size
     * Warning: Invalidates pointers and defeats growth optimization
     */
    void shrink_to_fit()
    {
        if (m_rows < m_buffer_rows || m_cols < m_buffer_cols)
        {
            std::vector<T> new_buffer(m_rows * m_cols);

            // Copy data to compact buffer
            for (size_t i = 0; i < m_rows; ++i)
            {
                if (std::is_trivially_copyable<T>::value)
                {
                    std::memcpy(&new_buffer[i * m_cols], &m_buffer[index(i, 0)], m_cols * sizeof(T));
                }
                else
                {
                    for (size_t j = 0; j < m_cols; ++j)
                    {
                        new_buffer[i * m_cols + j] = m_buffer[index(i, j)];
                    }
                }
            }

            m_buffer.swap(new_buffer);
            m_buffer_rows = m_rows;
            m_buffer_cols = m_cols;
        }
    }

    // Dimension getters
    size_t rows() const { return m_rows; }
    size_t cols() const  { return m_cols; }
    size_t stride() const { return m_buffer_cols; } // for numpy stride calculationx

    /**
     * Raw data access for numpy integration
     * Returns pointer to start of buffer
     * Note: stride() != cols() when buffer is over-allocated
     */
    T *get_data_ptr() { return m_buffer.data(); }
    const T *get_data_ptr() const { return m_buffer.data(); }

    // 2D element access ...
    inline T &operator()(size_t i, size_t j) { return m_buffer[index(i, j)]; }
    inline const T &operator()(size_t i, size_t j) const  { return m_buffer[index(i, j)]; }

    // Overloads for int indices for backward compatibility.
    inline T &operator()(int i, int j)  { return operator()(static_cast<size_t>(i), static_cast<size_t>(j)); }
    inline const T &operator()(int i, int j) const { return operator()(static_cast<size_t>(i), static_cast<size_t>(j)); }

    /**
     * @brief Returns a copy of row i as std::vector<T>.
     * @note This is a copy; for slicing without copy, consider returning a view.
     */
    std::vector<T> operator()(size_t i) const
    {
        std::vector<T> row(m_cols);
        for (size_t j = 0; j < m_cols; ++j)
        {
            row[j] = m_buffer[index(i, j)];
        }
        return row;
    }
};

#endif
