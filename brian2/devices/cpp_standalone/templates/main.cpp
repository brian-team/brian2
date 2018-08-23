#include <stdlib.h>
#include "objects.h"
#include <ctime>
#include <time.h>
{{ openmp_pragma('include') }}
#include "run.h"
#include "brianlib/common_math.h"
#include "randomkit.h"

{% for codeobj in code_objects | sort(attribute='name') %}
#include "code_objects/{{codeobj.name}}.h"
{% endfor %}

{% for name in user_headers | sort %}
#include {{name}}
{% endfor %}

#include <iostream>
#include <fstream>

{{report_func|autoindent}}

int main(int argc, char **argv)
{
    {% if openmp_pragma('with_openmp') %}
    double _clock_start;
    _clock_start = omp_get_wtime();
    {% else %}
    std::clock_t _clock_start;
    _clock_start = std::clock();
    {% endif %}

	brian_start();

	{
		using namespace brian;

		{{ openmp_pragma('set_num_threads') }}
        {{main_lines|autoindent}}
	}

	brian_end();

	{% if openmp_pragma('with_openmp') %}
    Network::_after_run_time = omp_get_wtime() - _clock_start;
    {% else %}
    Network::_after_run_time = ((double)(std::clock() - _clock_start) / CLOCKS_PER_SEC);
    {% endif %}
    // Write last run info to disk
	ofstream outfile_last_run_info;
	outfile_last_run_info.open("results/last_run_info.txt", ios::out);
	if(outfile_last_run_info.is_open())
	{
		outfile_last_run_info << (Network::_before_run_time) << " ";
		outfile_last_run_info << (Network::_last_run_time) << " ";
		outfile_last_run_info << (Network::_after_run_time) << " ";
		outfile_last_run_info << (Network::_last_run_completed_fraction) << std::endl;
		outfile_last_run_info.close();
	} else
	{
	    std::cout << "Error writing last run info to file." << std::endl;
	}
	return 0;
}
