
# Installation

## Create a conda environment

If you do not have conda, we recommend installing [miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/).

Then, in your command line tool:

1. Create a conda environment
    
    ```bash
    conda create -n 'napari-n2v' python=3.9
    conda activate napari-n2v
    ```
    
2. Follow the [TensorFlow installation step-by-step](https://www.tensorflow.org/install/pip#linux_1) for your 
operating system.
3. Install `napari`:
    ```bash
    pip install "napari[all]==0.4.15"
    ```


## Install napari-n2v

### Install napari-n2v through the napari-hub

<!---

Check-out the instructions on [installing plugins via the napari hub](https://napari.org/stable/plugins/find_and_install_plugin.html). 
This step is performed after [starting napari](#start-napari-n2v).
-->
(soon)

### Install napari-n2v via pip

Within the previously installed conda environment, type:

```bash
pip install git+https://github.com/juglab/napari-n2v.git
```

### Install napari-n2v from source

Clone the repository:
```bash
git clone https://github.com/juglab/napari-n2v.git
```

Navigate to the newly created folder:
```bash
cd napari-n2v
```

Within the previously installed conda environment, type:

```bash
pip install -e .
```

# Start napari-N2V

1. Using the terminal with the `napari-n2v` environment active, start napari:
    
    ```bash
    napari
    ```
    
2. Load one of the napari-N2V plugin.