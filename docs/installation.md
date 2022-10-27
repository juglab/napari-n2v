
# Installation

## Set up the conda environment

If you do not have conda, we recommend installing [miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/).

Then, in your command line tool:

1. Create a conda environment
    
    ```bash
    conda create -n napari-n2v python=3.9
    conda activate napari-n2v
    ```
    
2. Follow the [TensorFlow installation step-by-step](https://www.tensorflow.org/install/pip#linux_1) for your 
operating system. For macOS users, you might want to try [these instructions](https://developer.apple.com/metal/tensorflow-plugin/).
3. Install `napari`:
    ```bash
    pip install "napari[all]==0.4.15"
    ```

## Install napari-n2v

There are several ways to install `napari-n2v`: through the napari-hub, via pip or from source. We recommend installing via pip.


### Install napari-n2v via pip

Within the previously installed conda environment, type:

```bash
pip install napari-n2v
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