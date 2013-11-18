{% extends 'common_group.cpp' %}

{% block maincode %}
    // USES_VARIABLES { _synaptic_pre, _synaptic_post, rand}
	srand((unsigned int)time(NULL));
	int _buffer_size = 1024;
	int *_prebuf = new int[_buffer_size];
	int *_postbuf = new int[_buffer_size];
	int *_synprebuf = new int[1];
	int *_synpostbuf = new int[1];
	int _curbuf = 0;
	for(int i=0; i<_num_all_pre; i++)
	{
		for(int j=0; j<_num_all_post; j++)
		{
		    const int _vectorisation_idx = j;
			// Define the condition
		    {{ super() }}
			// Add to buffer
			if(_cond)
			{
			    if (_p != 1.0) {
			        // We have to use _rand instead of rand to use our rand
			        // function, not the one from the C standard library
			        if (_rand(_vectorisation_idx) >= _p)
			            continue;
			    }

			    for (int _repetition=0; _repetition<_n; _repetition++) {
                    _prebuf[_curbuf] = _pre_idcs;
                    _postbuf[_curbuf] = _post_idcs;
                    _curbuf++;
                    // Flush buffer
                    if(_curbuf==_buffer_size)
                    {
                        _flush_buffer(_prebuf, _synaptic_pre_object, _curbuf);
                        _flush_buffer(_postbuf, _synaptic_post_object, _curbuf);
                        _curbuf = 0;
                    }
                }
			}
		}
	}
	// Final buffer flush
	_flush_buffer(_prebuf, _synaptic_pre_object, _curbuf);
	_flush_buffer(_postbuf, _synaptic_post_object, _curbuf);
	delete [] _prebuf;
	delete [] _postbuf;
	delete [] _synprebuf;
	delete [] _synpostbuf;
{% endblock %}

{% block support_code_block %}
// Flush a buffered segment into a dynamic array
void _flush_buffer(int *buf, py::object &dynarr, int N)
{
	int _curlen = dynarr.attr("shape")[0];
	int _newlen = _curlen+N;
	// Resize the array
	py::tuple _newlen_tuple(1);
	_newlen_tuple[0] = _newlen;
	dynarr.mcall("resize", _newlen_tuple);
	// Get the potentially newly created underlying data arrays
	int *data = (int*)(((PyArrayObject*)(PyObject*)dynarr.attr("data"))->data);
	// Copy the values across
	for(int i=0; i<N; i++)
	{
		data[_curlen+i] = buf[i];
	}
}

{{ super() }}

{% endblock %}
