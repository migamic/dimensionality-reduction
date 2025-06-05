# Scalable Force Scheme (SFS)

Implementation of Scalable Force Scheme (+ Force Scheme (FS) and Gradient Force Scheme (GFS)) and experiments to test its performance.

## Quick test
1. Clone this repository
2. `pip install -r requirements.txt`
3. `python test_fs.py`

This will run SFS with a variety of test datasets. Check `test_fs.py` code to explore other options, such as trying other datasets, computing/disabling metrics, tweaking hyperparameters, etc.


## Replicating the results of the paper

The script `run_batch.py` is used to create a set of results relevant for comparing FS, GFS and SFS. At the beginning of that file you can see the different hyperparameter setups used to differentiate between FS, GFS and SFS.

Make sure the data is present at the `data` directory (extracting the `data.zip` file).


## Running SFS on your project

The code for SFS is in `dr/force_scheme.py`. The `ForceScheme` class accepts several arguments, allowing the user to choose between FS, GFS, SFS, or other alternative setups. Examples of hyperparameter setups for FS, GFS, and SFS can be found at the beginning of `run_batch.py`, but we recommend adapting the hyperparameters to every particular use case.

You might find the code in `test_fs.py` useful as an example on how to run SFS and process the results.

If you find this paper or implementation useful, please consider citing our [EuroVA paper](https://diglib.eg.org/items/c29bb6a2-7bf6-4daf-9018-912886c7c000)!
```{tex}
@inproceedings{10.2312:eurova.20251098,
  booktitle = {EuroVis Workshop on Visual Analytics (EuroVA)},
  editor = {Schulz, Hans-Jörg and Villanova, Anna},
  title = {{Scalable Force Scheme: a fast method for projecting large datasets}},
  author = {Ros, Jaume and Arleo, Alessio and Paulovich, Fernando V.},
  year = {2025},
  publisher = {The Eurographics Association},
  ISSN = {2664-4487},
  ISBN = {978-3-03868-283-7},
  DOI = {10.2312/eurova.20251098}
}
```
