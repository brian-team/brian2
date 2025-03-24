{# USES_VARIABLES { _spikespace, neuron_index, _timebins, _period_bins, _lastindex, N, _spike_time, dt, period } #}
{% extends 'common_group.cpp' %}

{% block before_code %}
    // Copy of the SpikeGeneratorGroup.before_run code
    const double _dt = {{dt.item()}};  // Always copy dt
    const double _period = {{period_}};  // Always copy period

    // Always recalculate _timesteps
    std::vector<int32_t> _timesteps({{_spike_time}}.size());
    for (size_t i = 0; i < _timesteps.size(); i++) {
        _timesteps[i] = static_cast<int32_t>({{_spike_time}}[i] / _dt);
    }

    // Get current simulation time from Brian 2's clock instead of 't'
    extern double defaultclock_t;  
    const int32_t _current_step = static_cast<int32_t>(defaultclock_t / _dt);

    // Always update _lastindex
    int32_t _last_idx = 0;
    for (size_t i = 0; i < _timesteps.size(); i++) {
        if (_timesteps[i] < _current_step) {
            _last_idx = i + 1;
        } else {
            break;
        }
    }
    {{_lastindex}} = _last_idx;

    // Always recalculate _timebins
    const double _shift = 1e-3 * _dt;
    std::vector<int32_t> _timebins({{_spike_time}}.size());
    for (size_t i = 0; i < _timebins.size(); i++) {
        _timebins[i] = static_cast<int32_t>({{_spike_time}}[i] + _shift) / _dt;
    }
    {{_timebins}} = _timebins;

    // Always recalculate _period_bins (no checks)
    {{_period_bins}} = static_cast<int32_t>(std::round(_period / _dt));
{% endblock %}

{% block maincode %}

    const int32_t _the_period = {{_period_bins}};
    int32_t _timebin = static_cast<int32_t>(defaultclock_t / {{dt.item()}});  // Use Brian 2's clock instead of 't_in_timesteps'

    // Always recalculate timebin with period
    _timebin %= _the_period;
    {{_lastindex}} = 0;

    int32_t _cpp_numspikes = 0;

    for (size_t _idx = {{_lastindex}}; _idx < _num_timebins; _idx++) {
        if ({{_timebins}}[_idx] > _timebin)
            break;

        {{_spikespace}}[_cpp_numspikes++] = {{neuron_index}}[_idx];
    }

    {{_spikespace}}[N] = _cpp_numspikes;

    {{_lastindex}} += _cpp_numspikes;

{% endblock %}
