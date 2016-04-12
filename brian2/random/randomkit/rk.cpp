#include "rk.h"

rk_state *internal_state = NULL;

rk_state **get_rk_state()
{
    return &internal_state;
}