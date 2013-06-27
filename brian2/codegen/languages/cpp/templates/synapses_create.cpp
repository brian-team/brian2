{% macro main() %}
	srand((unsigned int)time(NULL));
	int _buffer_size = 1024;
	int *_prebuf = new int[_buffer_size];
	int *_postbuf = new int[_buffer_size];
	int _curbuf = 0;
	for(int _source_neuron_idx=0; _source_neuron_idx<_num_source_neurons; _source_neuron_idx++)
	{
		for(int _target_neuron_idx=0; _target_neuron_idx<_num_target_neurons; _target_neuron_idx++)
		{
			// Define the condition
			{% for line in code_lines %}
			{{line}}
			{% endfor %}
			// Add to buffer
			if(_cond)
			{
				_prebuf[_curbuf] = _source_neuron_idx;
				_postbuf[_curbuf] = _target_neuron_idx;
				_curbuf++;
			}
			// Flush buffer
			if(_curbuf==_buffer_size)
			{
				_flush_buffer(_prebuf, _presynaptic, _curbuf);
				_flush_buffer(_postbuf, _postsynaptic, _curbuf);
				_curbuf = 0;
			}
		}

	}
	// Final buffer flush
	_flush_buffer(_prebuf, _presynaptic, _curbuf);
	_flush_buffer(_postbuf, _postsynaptic, _curbuf);
	delete [] _prebuf;
	delete [] _postbuf;
{% endmacro %}

{% macro support_code() %}
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
double _rand(int n)
{
	return (double)rand()/RAND_MAX;
}
{% endmacro %}
