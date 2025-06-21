# Scalable Force Scheme (SFS)

Implementation of Scalable Force Scheme (+ Force Scheme (FS) and Gradient Force Scheme (GFS)) and experiments to test its performance.

If you find this method or implementation useful, please consider citing our [EuroVA paper](https://diglib.eg.org/items/c29bb6a2-7bf6-4daf-9018-912886c7c000).
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


## Quick test
1. Clone this repository
2. `pip install -r requirements.txt`
3. `python test_fs.py`

This will run SFS with a variety of test datasets. Check `test_fs.py` code to explore other options, such as trying other datasets, computing/disabling metrics, tweaking hyperparameters, etc.


## Running FS, GFS and SFS on your project

```python
import numpy as np
import matplotlib.pyplot as plt
from force_scheme import ForceScheme, GFS, SFS

X = np.load('data/coil20/X.npy')       # Example data. X should be an NxD array
X_2D = ForceScheme().fit_transform(X)  # For Force-Scheme
X_2D = GFS().fit_transform(X)          # For Gradient Force-Scheme
X_2D = SFS().fit_transform(X)          # For Scalable Force-Scheme

# X_2D is an Nx2 array of the 2D embeddings for the N points

plt.scatter(X_2D[:,0], X_2D[:,1])
plt.show()
```

> Note that all the functionality of GFS and SFS has been implemented in the `ForceScheme` class. The `GFS` and `SFS` classes are simply wrappers of this class with different default values for hyperparameters.

You might find the code in `test_fs.py` useful as an example on how to run SFS and process the results.


## Replicating the results of the paper

The script `run_batch.py` is used to create a set of results relevant for comparing FS, GFS and SFS. At the beginning of that file you can see the different hyperparameter setups used to differentiate between FS, GFS and SFS.

Make sure the data is present at the `data` directory (extracting the `data.zip` file).
