'''
Module implementing the "example" device. This is a device for documentation
purposes that is not useful in practice. It will write out a Python script that
should perform the same simulation as the script simulated with the example
device.
'''
import os
import shutil
import subprocess
import sys
import inspect
import platform
from collections import defaultdict, Counter
import numbers
import tempfile
from distutils import ccompiler

import numpy as np
from cpuinfo import cpuinfo

import brian2

from brian2.codegen.cpp_prefs import get_compiler_and_args
from brian2.core.network import Network
from brian2.devices.device import Device, all_devices, set_device, reset_device
from brian2.core.variables import *
from brian2.core.namespace import get_local_namespace
from brian2.parsing.rendering import CPPNodeRenderer
from brian2.synapses.synapses import Synapses
from brian2.core.preferences import prefs, BrianPreference
from brian2.utils.filetools import copy_directory, ensure_directory, in_directory
from brian2.utils.stringtools import word_substitute
from brian2.codegen.generators.cpp_generator import c_data_type
from brian2.units.fundamentalunits import Quantity, have_same_dimensions
from brian2.units import second, ms
from brian2.utils.logger import get_logger, std_silent

from .codeobject import CPPStandaloneCodeObject, openmp_pragma


__all__ = []

logger = get_logger(__name__)

class ExampleDevice(Device):
    '''
    The `Device` used for C++ standalone simulations.
    '''
    def __init__(self):
        super(ExampleDevice, self).__init__()
        

example_device = ExampleDevice()
all_devices['example'] = example_device
