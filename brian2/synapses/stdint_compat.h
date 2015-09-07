// Work around the fact that older MSVC versions don't have stdint.h
#ifdef _MSC_VER
typedef __int32 int32_t;
typedef __int64 int64_t;
typedef unsigned __int64 uint64_t;
#else
#include <stdint.h>
#endif
