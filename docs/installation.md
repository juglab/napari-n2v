
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
3. Install `napari` and `napari-n2v`:
   ```bash
      pip install "napari[all]" napari-n2v
   ```

> Note: napari-n2v was tested with TensorFlow 2.10 (cuda 11.2 and cudnn 8.1) and
TensorFlow 2.13 (cuda 11.8 and cudnn 8.6) on a Linux machine (NVIDIA A40-16Q GPU).

> **Important**: In order to access the GPU with Tensorflow, it is necessary to
> export the CUDA library path in your conda environment. Installation 
> instructions on the TensorFlow website do just that. 

> For TF 2.10, we recommand running the following in your environment:
> ```bash
> mkdir -p $CONDA_PREFIX/etc/conda/activate.d
> echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
> ```

> If you encounter the following problem with TF 2.13: "DNN library is not found", you
> can try to run in your environment:
> ```bash
> CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
> export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
> ```
   
## macOS

> Note: These instructions are for GPU support. Apple's tensorflow-metal is only officially supported for macOS 12 and 
> higher. For CPU, you can try the Follow the [TensorFlow instructions](https://www.tensorflow.org/install/pip#macos_1) 

1. Set up env with napari and pyqt5
   ```bash
      conda create -n napari-n2v -c conda-forge python=3.9 pyqt imagecodecs napari
   ```
2. Install tensorflow following [Apple's instructions](https://developer.apple.com/metal/tensorflow-plugin/)
3. Install napari-n2v
   ```bash
      pip install napari-n2v
   ```

# Start napari-N2V

1. Using the terminal with the `napari-n2v` environment active, start napari:
    
    ```bash
    napari
    ```
    
2. Load one of the napari-N2V plugin.
