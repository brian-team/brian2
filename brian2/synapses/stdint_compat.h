#ifndef _BRIAN_STDINT_COMPAT_H
#define _BRIAN_STDINT_COMPAT_H
// Work around the fact that older MSVC versions don't have stdint.h
#ifdef _MSC_VER
typedef __int32 int32_t;
typedef __int64 int64_t;
typedef unsigned __int64 uint64_t;
#else
#include <stdint.h>
#endif
// Implement the int_ function here so that it can also be used from Cython
template<typename T> inline int int_(T value)
{
    return (int)value;
}
template<> inline int int_(bool value)
{
    return value ? 1 : 0;
}
#endif
