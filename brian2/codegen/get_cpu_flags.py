"""
This script is used to ask for the CPU flags on Windows. We use this instead of
importing the cpuinfo package, because recent versions of py-cpuinfo use the
multiprocessing module, and any import of cpuinfo that is not within a
`if __name__ == '__main__':` block will lead to the script being executed twice.

The CPU flags are printed to stdout encoded as JSON.
"""


import json

if __name__ == "__main__":
    import cpuinfo

    flags = cpuinfo.get_cpu_info()["flags"]
    print(json.dumps(flags))
