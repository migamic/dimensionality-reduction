# Scalable Force Scheme (SFS)

Implementation of Scalable Force Scheme (+ Force Scheme (FS) and Gradient Force Scheme (GFS)) and experiments to test its performance.

## How to run

The code for the DR method is in `dr/force_scheme.py`. Check the code at `test_fs.py` for an example on how to run it, as well as some utilities used to report the results.

The script `run_batch.py` is used to create a set of results relevant for comparing FS, GFS and SFS. At the beginning of that code you can see the different hyperparameter setups used to differentiate between FS, GFS and SFS.
