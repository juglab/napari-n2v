# Useful tips

## My GPU is not found

If your GPU is not found, you might want to try to force export the path to cuda in your conda environment:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
```

Otherwise, try following the [TensorFlow guidelines](https://www.tensorflow.org/install/pip).

## Training stopped before the end

If the number of steps is too high for the batch size, TensorFlow will stop the training. The plugin will consider training to be done, while the number of epochs and steps has not been completed.

The number of steps should be chosen to be roughly equal to the number of patches divided by the batch size. Unfortunately, we do not know the number of patches before starting training, as it depends on multiple factor (size of the images, augmentation, patch size).

A way around it is to look at console output at the beginning of the training and catch the number of training patches printed out. Example: 2400 patches if the size of the training set is (2400, 64, 64, 1).

Start the training again (don’t forget to reset the model or to restart the plugin), albeit with the corrected steps.

## Getting out-of-memory (OOM) errors

If you have OOM errors, first determine whether this is physical, virtual or GPU memory from the error message. In most cases, this will be a GPU memory problem as GPU tend to be the limiting factor.

Few parameters allow reducing the load on the GPU memory:

- `Bacth size`: reduce the batch size to improve the chances to fit in the GPU memory.
- `Patch XY` and `Patch Z`: reduce the patch size. Note that patch size might be important to allow the network to see enough features of the image. Do no go smaller than the features of your images.
- Tiling: during `prediction` tiling helps breaking down the images into tiles in order to fit in the GPU memory.