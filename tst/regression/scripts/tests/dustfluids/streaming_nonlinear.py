"""
Example test script.

This is a complete, working example that can be run as part of the test suite. It does a
simple test of a relativistic shock tube using the GR framework. There are many comments
in order to make this file self-explanatory, but the actual working code is only 28
lines long.

There are three functions defined here:
    prepare(**kwargs)
    run(**kwargs)
    analyze()
All three must be defined with the same names and no required inputs in order to make a
working script. They are called in sequence from the main test script run_tests.py.
Additional support functions can be defined here, to be called by the three primary fns.

Heavy use is made of support utilities defined in scripts/utils/athena.py. These are
general-purpose Python scripts that interact with Athena++. They should be used whenever
possible, since they work together to compile and run Athena++ and read the output data.
In particular, proper use of them will result in all files outside tst/regression/ being
in the same state after the test as they were before (including whatever configured
version of Athena++ existed in athena/bin/), as well as cleaning up any new files
produced by the test.
"""

# Modules
import numpy as np                             # standard Python module for numerics
import sys                                     # standard Python module to change path
import scripts.utils.athena as athena          # utilities for running Athena++
import scripts.utils.comparison as comparison  # more utilities explicitly for testing
sys.path.insert(0, '../../vis/python')         # insert path to Python read scripts
import athena_read                             # utilities for reading Athena++ data # noqa
athena_read.check_nan_flag = True              # raise exception when encountering NaNs


def prepare(**kwargs):
    """
    Configure and make the executable.

    This function is called first. It is responsible for calling the configure script and
    make to create an executable. It takes no inputs and produces no outputs.
    """

    # Configure as though we ran
    #     python configure.py --prob=streaming_eigen --ndustfluids=1 eos=isothermal nghost=3
    athena.configure(prob='streaming_nonlinear',
                     ndustfluids='1',
                     eos='isothermal',
                     nghost='3',
                     flux='roe',
                     **kwargs)

    # Call make as though we ran
    #     make clean
    #     make
    # from the athena/ directory.
    athena.make()


def run(**kwargs):
    """
    Run the executable.

    This function is called second. It is responsible for calling the Athena++ binary in
    such a way as to produce testable output. It takes no inputs and produces no outputs.
    """
    arguments = ['time/tlim=62.84',
                 'output2/file_type=vtk',
                 'mesh/nx1=64',
                 'mesh/nx2=64']

    athena.run('dust/athinput.streaming_RunBA', arguments)
    return


def analyze():
    """
    Analyze the output and determine if the test passes.

    This function is called third; nothing from this file is called after it. It is
    responsible for reading whatever data it needs and making a judgment about whether or
    not the test passes. It takes no inputs. Output should be True (test passes) or False
    (test fails).
    """

    analyze_status = True

    return analyze_status
