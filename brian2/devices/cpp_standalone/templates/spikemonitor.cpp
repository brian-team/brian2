{% extends 'common_group.cpp' %}

{% block maincode %}
	//// MAIN CODE ////////////
    // USES_VARIABLES { _t, _i, t, _spikespace, _count,
    //                  _source_start, _source_stop}
	int _num_spikes = {{_spikespace}}[_num_spikespace-1];
    if (_num_spikes > 0)
    {
        int _start_idx = 0;
        int _end_idx = - 1;
        for(int _j=0; _j<_num_spikes; _j++)
        {
            const int _idx = {{_spikespace}}[_j];
            if (_idx >= _source_start) {
                _start_idx = _j;
                break;
            }
        }
        for(int _j=_start_idx; _j<_num_spikes; _j++)
        {
            const int _idx = {{_spikespace}}[_j];
            if (_idx >= _source_stop) {
                _end_idx = _j;
                break;
            }
        }
        if (_end_idx == -1)
            _end_idx =_num_spikes;
        _num_spikes = _end_idx - _start_idx;
        if (_num_spikes > 0) {
        	for(int _j=_start_idx; _j<_end_idx; _j++)
        	{
        		const int _idx = {{_spikespace}}[_j];
        		{{_object__i}}.push_back(_idx-_source_start);
        		{{_object__t}}.push_back(t);
        	}
        }
    }
{% endblock %}

{% block extra_functions_cpp %}
void _write_{{codeobj_name}}()
{
	ofstream outfile_t;
	outfile_t.open("results/{{codeobj_name}}_t", ios::binary | ios::out);
	if(outfile_t.is_open())
	{
		outfile_t.write(reinterpret_cast<char*>(&{{_object__t}}[0]), {{_object__t}}.size()*sizeof({{_object__t}}[0]));
		outfile_t.close();
	} else
	{
		cout << "Error writing output file results/{{codeobj_name}}_t." << endl;
	}
	ofstream outfile_i;
	outfile_i.open("results/{{codeobj_name}}_i", ios::binary | ios::out);
	if(outfile_i.is_open())
	{
		outfile_i.write(reinterpret_cast<char*>(&{{_object__i}}[0]), {{_object__i}}.size()*sizeof({{_object__i}}[0]));
		outfile_i.close();
	} else
	{
		cout << "Error writing output file results/{{codeobj_name}}_i." << endl;
	}
//	ofstream outfile;
//	outfile.open("results/{{codeobj_name}}.txt", ios::out);
//	if(outfile.is_open())
//	{
//		for(int s=0; s<{{_i}}.size(); s++)
//		{
//			outfile << {{_i}}[s] << ", " << {{_t}}[s] << endl;
//		}
//		outfile.close();
//	} else
//	{
//		cout << "Error writing output file." << endl;
//	}
}

void _debugmsg_{{codeobj_name}}()
{
	cout << "Number of spikes: " << {{_object__i}}.size() << endl;
}
{% endblock %}

{% block extra_functions_h %}
void _write_{{codeobj_name}}();
void _debugmsg_{{codeobj_name}}();
{% endblock %}

{% macro main_finalise() %}
_write_{{codeobj_name}}();
_debugmsg_{{codeobj_name}}();
{% endmacro %}
