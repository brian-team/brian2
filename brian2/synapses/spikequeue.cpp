#include "spikequeue.h"

// Operator to serialize the spike queue state to an output stream
ostream& operator<<(ostream& os, const CSpikeQueue& queue)
{
    const auto state = queue._full_state();
    os << state.first << "\n";
    os << state.second.size() << "\n";
    for (const auto& inner_vec : state.second)
    {
        os << inner_vec.size() << "\n";
        if (inner_vec.size() > 0) {
            for (const auto& val : inner_vec)
                os << val << " ";

            os << "\n";
        }
    }
    return os;
}

// Operator to deserialize the spike queue state from an input stream
istream& operator>>(istream& is, CSpikeQueue& queue)
{
    int stored_offset;
    size_t outer_size;

    is >> stored_offset;
    is >> outer_size;
    vector<vector<int32_t>> stored_queue(outer_size);

    for (size_t i = 0; i < outer_size; i++)
    {
        size_t inner_size;
        is >> inner_size;
        stored_queue[i].resize(inner_size);

        for (size_t j = 0; j < inner_size; j++)
            is >> stored_queue[i][j];
    }

    pair<int, vector<vector<int32_t>>> state(stored_offset, stored_queue);
    queue._restore_from_full_state(state);

    return is;
}
