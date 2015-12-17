#ifndef _BRIAN_COMMON_MATH_H
#define _BRIAN_COMMON_MATH_H

#include<limits>
#include<stdlib.h>

#ifdef _MSC_VER
#define INFINITY (std::numeric_limits<double>::infinity())
#define NAN (std::numeric_limits<double>::quiet_NaN())
#define M_PI 3.14159265358979323846
#endif

#endif
