
# Installation

## Create a conda environment

If you do not have conda, we recommend installing [miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/).

Then, in your command line tool:

1. Create a conda environment
    
    ```bash
    conda create -n 'napari-n2v' python=3.9
    conda activate napari-n2v
    ```
    
2. Install the packages necessary to use GPU-acceleration
    
    ```bash
    conda install cudatoolkit=11.2 cudnn git -c conda-forge
    pip install tensorflow==2.5
    ```
    
3. Install the latest version of N2V
    
    ```bash
    pip install git+https://github.com/juglab/n2v.git
    ```
    
4. Install napari
    
    ```bash
    pip install "napari[all]"
    ```
   
   Note that `pip` might complain about `typing-extensions` versions. We will apply a duck-tape patch in the
   next point. This should not impact the functioning of the plugin.
    
5. Fix the typing-extensions version
    
    ```bash
    pip uninstall typing-extensions  # remove 3.7.4
    pip install typing-extensions    # install 4.3.0
    ```
    

## Install napari-n2v through the napari-hub

## Install napari-n2v from source

```bash
pip install git+https://github.com/juglab/napari-n2v.git
```

# Start napari-N2V

1. Using the terminal with the `napari-n2v` environment active, start napari:
    
    ```bash
    napari
    ```
    
2. Load one of the napari-N2V plugin