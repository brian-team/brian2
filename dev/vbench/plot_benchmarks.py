import tempfile
from datetime import date
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter, MonthLocator

from run_benchmarks import benchmarks

DB_PATH = '/tmp/vbench_test/benchmarks.db'

for benchmark in benchmarks:
    # prepare the graph
    print '\tCreating graph'
    tmp_path = tempfile.mktemp(suffix='.png')
    results = benchmark.get_results(DB_PATH)['timing']

    if len(results) == 0:
        continue

    # make basic plot with grid in background
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axis_bgcolor('#eeeeec')
    if results.max() > 2500:
        divider = 1000.
        unit = 's'
    else:
        divider = 1.
        unit = 'ms'

    ax.plot(results.index, results / divider, color='#2e3436', linewidth=1.5)
    max_y = ax.get_ylim()[1]
    ax.set_ylim(0, max_y)
    ax.grid(True, color='white', linestyle='-', linewidth=3)
    ax.set_axisbelow(True)

    # Disable spines.
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Disable ticks.
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # format dates
    formatter = DateFormatter("%m/%Y")
    ax.xaxis.set_major_locator(MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.xaxis.set_major_formatter(formatter)
    fig.autofmt_xdate(rotation=45)

    ax.set_title(benchmark.name)
    ax.set_xlabel('Date')
    ax.set_ylabel('Run time (%s)' % unit)

    plt.tight_layout()
    plt.show()