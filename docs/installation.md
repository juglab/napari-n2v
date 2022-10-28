
# Installation

## Windows/Linux

If you do not have conda, we recommend installing [miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/).

1. Then, in your command line tool create a conda environment 
   ```bash
       conda create -n napari-n2v python=3.9
       conda activate napari-n2v
   ```
2. Follow the [TensorFlow installation step-by-step](https://www.tensorflow.org/install/pip#linux_1) for your 
operating system.
3. Install `napari`:
   ```bash
      pip install "napari[all]==0.4.15"
   ```
4. Install `napari-n2v`:
   ```bash
      pip install napari-n2v
   ```
   
## macOS

## arm64 (Apple Silicon M1)

1. Set up env with napari and pyqt5
   ```bash
      conda create -n napari-n2v -c conda-forge python=3.9 napari pyqt5 imagecodecs
   ```
2. Install tensorflow following [your link](https://developer.apple.com/metal/tensorflow-plugin/)
3. Install napari-n2v
   ```bash
      pip install napari-n2v
   ```

## x86 (Intel)

# Start napari-N2V

1. Using the terminal with the `napari-n2v` environment active, start napari:
    
    ```bash
    napari
    ```
    
2. Load one of the napari-N2V plugin.