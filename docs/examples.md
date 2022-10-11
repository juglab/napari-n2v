# Example pipelines

The plugins come with sample data that can be loaded into napari using `File/Open sample/napari-n2v`. As the images are downloaded from a remote server, the process can seem idle for a while before eventually loading the images as napari layers.

In this section, we describe how to reproduce the results from the N2V Github repository using the napari plugins.

> **Important note**: if you are using a GPU with little memory (e.g. 4 GB), then most of the shown 
> settings will not work because the batches will probably not fit in memory. Try reducing the batch
> size while increasing the number of steps. This will obviously increase the running time.

## 2D BSD68

The [example notebook](https://github.com/juglab/n2v/blob/master/examples/2D/denoising2D_BSD68/BSD68_reproducibility.ipynb) generates a configuration containing all the parameters used for training and reproducing the results in the N2VConfig call:

```bash
config = N2VConfig(X, 
                   unet_kern_size=3, 
                   train_steps_per_epoch=400, 
                   train_epochs=200, 
                   train_loss='mse', 
                   batch_norm=True, 
                   train_batch_size=128, 
                   n2v_perc_pix=0.198, 
                   n2v_patch_shape=(64, 64), 
                   unet_n_first = 96,
                   unet_residual = True,
                   n2v_manipulator='uniform_withCP', 
                   n2v_neighborhood_radius=2,
                   single_net_per_channel=False)
```

The resulting configuration is:

```bash
{'means': ['110.72957232412905'], 
 'stds': ['63.656060106500874'],
 'n_dim': 2,
 'axes': 'YXC',
 'n_channel_in': 1,
 'n_channel_out': 1,
 'unet_residual': True, # Expert settings / U-Net residuals
 'unet_n_depth': 2,   # Expert settings / U-Net depth
 'unet_kern_size': 3, # Expert settings / U-Net kernel size
 'unet_n_first': 96,  # Expert settings / U-Net n filters
 'unet_last_activation': 'linear',
 'unet_input_shape': (None, None, 1),
 'train_loss': 'mse', # Expert settings / Train loss
 'train_epochs': 200,  # N epochs
 'train_steps_per_epoch': 400, # N steps
 'train_learning_rate': 0.0004, # Expert settings / Learning rate
 'train_batch_size': 128, # Batch size 
 'train_tensorboard': True,
 'train_checkpoint': 'weights_best.h5',
 'train_reduce_lr': {'factor': 0.5, 'patience': 10},
 'batch_norm': True,
 'n2v_perc_pix': 0.198, # Expert settings / N2V pixel %
 'n2v_patch_shape': (64, 64), # Patch XY (and Patch Z)
 'n2v_manipulator': 'uniform_withCP', # Expert settings / N2V manipulator
 'n2v_neighborhood_radius': 2,  # Expert settings / N2V radius
 'single_net_per_channel': False, # Expert settings / Split channels
 'structN2Vmask': None, # Expert settings / structN2V
 'probabilistic': False}
```

Here we commented some lines with the equivalent parameters in the napari plugin. Parameters that were not specifically set in the `N2VConfig` call are set their default and might not need to be set in the napari plugin either.

In order to reproduce the result using the plugin, we then follow these steps:

1. In napari, go to `File / Open sample / napari-n2v / Download data (2D)`, after the time necessary to download the data, it will automatically add the BSD68 data set to napari.
2. Confirm that your environment is properly set for GPU training by checking that the GPU indicator (top right) in the plugin displays a greenish GPU label.
3. Select the validation layer in `Val`.
4. In `Training parameters`, set: <br>
`N epochs` = 200 <br>
`N steps` = 400 <br>
`Batch size` = 128 <br>
`Patch XY` = 64 <br>
5. Click on the gear button to open the `Expert settings` and set: <br>
`U-Net kernel size` = 3 <br>
`U-Net residuals` = True (check) <br>
`Split channels` = False (uncheck) <br>
`N2V radius` = 2 <br>
6. You can compare the configuration above to the rest of the `Expert settings` to confirm that the other default values are properly set.
7. Train!

If your GPU is too small for the training parameters (loading batches in the GPU memory creates out-of-memory errors), then you should decrease the `Batch size` parameter.

## 2D RGB example

The RGB notebook example can be found [here](https://github.com/juglab/n2v/blob/master/examples/2D/denoising2D_RGB/01_training.ipynb). 

```bash
config = N2VConfig(X, 
                   unet_kern_size=3, 
                   unet_n_first=64, 
                   unet_n_depth=3, 
                   train_steps_per_epoch=39, 
                   train_epochs=25, 
                   train_loss='mse', 
                   batch_norm=True, 
                   train_batch_size=128, 
                   n2v_perc_pix=0.198, 
                   n2v_patch_shape=(64, 64), 
                   n2v_manipulator='uniform_withCP', 
                   n2v_neighborhood_radius=5, 
                   single_net_per_channel=False)
```

- Complete configuration
    
    ```bash
    {'means': ['0.5511132', '0.59339416', '0.5724706'],
     'stds': ['0.2939913', '0.30135363', '0.3155007'],
     'n_dim': 2,
     'axes': 'YXC',
     'n_channel_in': 3,
     'n_channel_out': 3,
     'unet_residual': False,
     'unet_n_depth': 3,
     'unet_kern_size': 3,
     'unet_n_first': 64,
     'unet_last_activation': 'linear',
     'unet_input_shape': (None, None, 3),
     'train_loss': 'mse',
     'train_epochs': 25,
     'train_steps_per_epoch': 39,
     'train_learning_rate': 0.0004,
     'train_batch_size': 128,
     'train_tensorboard': True,
     'train_checkpoint': 'weights_best.h5',
     'train_reduce_lr': {'factor': 0.5, 'patience': 10},
     'batch_norm': True,
     'n2v_perc_pix': 0.198,
     'n2v_patch_shape': (64, 64),
     'n2v_manipulator': 'uniform_withCP',
     'n2v_neighborhood_radius': 5,
     'single_net_per_channel': False,
     'structN2Vmask': None,
     'probabilistic': False}
    ```
    

In order to reproduce the result using the plugin, we then follow these steps:

1. In napari, go to `File / Open sample / napari-n2v / Download data (RGB).`
2. Confirm that your environment is properly set for GPU training by checking that the GPU indicator (top right) in the plugin displays a greenish GPU label.
3. Make sure to enter `YXC` in `Axes`.
4. In `Training parameters`, set: <br>
`N epochs` = 25 <br>
`N steps` = 39 <br>
`Batch size` = 128 <br>
`Patch XY` = 64 <br>
5. Click on the gear button to open the `Expert settings` and set: <br>
`U-Net depth` = 3 <br>
`U-Net kernel size` = 3 <br>
`U-Net n filters` = 64 <br>
`Split channels` = False (uncheck)
6. You can compare the configuration above to the rest of the `Expert settings` to confirm that the other default values are properly set.
7. Train!
8. Note that for the prediction, you will probably need to use tiling.

## 2D SEM

The example notebook can be found [here](https://github.com/juglab/n2v/blob/master/examples/2D/denoising2D_SEM/01_training.ipynb).

```bash
config = N2VConfig(X, 
                   unet_kern_size=3, 
                   train_steps_per_epoch=27, 
                   train_epochs=20, 
                   train_loss='mse', 
                   batch_norm=True, 
                   train_batch_size=128, 
                   n2v_perc_pix=0.198, 
                   n2v_patch_shape=(64, 64), 
                   n2v_manipulator='uniform_withCP', 
                   n2v_neighborhood_radius=5)
```

- Complete configuration
    
    ```bash
    {'means': ['39137.844'],
     'stds': ['18713.77'],
     'n_dim': 2,
     'axes': 'YXC',
     'n_channel_in': 1,
     'n_channel_out': 1,
     'unet_residual': False,
     'unet_n_depth': 2,
     'unet_kern_size': 3,
     'unet_n_first': 32,
     'unet_last_activation': 'linear',
     'unet_input_shape': (None, None, 1),
     'train_loss': 'mse',
     'train_epochs': 20,
     'train_steps_per_epoch': 27,
     'train_learning_rate': 0.0004,
     'train_batch_size': 128,
     'train_tensorboard': True,
     'train_checkpoint': 'weights_best.h5',
     'train_reduce_lr': {'factor': 0.5, 'patience': 10},
     'batch_norm': True,
     'n2v_perc_pix': 0.198,
     'n2v_patch_shape': (64, 64),
     'n2v_manipulator': 'uniform_withCP',
     'n2v_neighborhood_radius': 5,
     'single_net_per_channel': True,
     'structN2Vmask': None,
     'probabilistic': False}
    ```
    

In order to reproduce the result using the plugin, we then follow these steps:

1. Download the [data](https://download.fht.org/jug/n2v/SEM.zip) and unzip it. Place the train and validation images in two different folders.
2. Start the plugin and point towards the train and validation folders in the `From disk`.
3. Confirm that your environment is properly set for GPU training by checking that the GPU indicator (top right) in the plugin displays a greenish GPU label.
4. Make sure to enter `YX` in `Axes`.
5. In `Training parameters`, set: <br>
`N epochs` = 20 <br>
`N steps` = 27 <br>
`Batch size` = 128 <br>
`Patch XY` = 64 <br>
6. Click on the gear button to open the `Expert settings` and set: <br>
`U-Net kernel size` = 3 <br>
7. You can compare the configuration above to the rest of the `Expert settings` to confirm that the other default values are properly set.
8. Train!
9. Note that for the prediction, you will probably need to use tiling.


## 2D SEM with N2V2

The example notebook can be found [here](https://github.com/juglab/n2v/blob/master/examples/2D/denoising2D_SEM/01_training.ipynb).

```bash
config = N2VConfig(X, 
                   unet_kern_size=3, 
                   train_steps_per_epoch=27, 
                   train_epochs=20, 
                   train_loss='mse', 
                   batch_norm=True, 
                   train_batch_size=128, 
                   n2v_perc_pix=0.198, 
                   n2v_patch_shape=(64, 64), 
                   n2v_manipulator='mean', 
                   n2v_neighborhood_radius=5,
                   blurpool=True,
                   skip_skipone=True,
                   unet_residual=False)
```

- Complete configuration
    
    ```bash
    {'means': ['39137.844'],
     'stds': ['18713.77'],
     'n_dim': 2,
     'axes': 'YXC',
     'n_channel_in': 1,
     'n_channel_out': 1,
     'unet_residual': False,
     'unet_n_depth': 2,
     'unet_kern_size': 3,
     'unet_n_first': 32,
     'unet_last_activation': 'linear',
     'unet_input_shape': (None, None, 1),
     'train_loss': 'mse',
     'train_epochs': 20,
     'train_steps_per_epoch': 27,
     'train_learning_rate': 0.0004,
     'train_batch_size': 128,
     'train_tensorboard': True,
     'train_checkpoint': 'weights_best.h5',
     'train_reduce_lr': {'factor': 0.5, 'patience': 10},
     'batch_norm': True,
     'n2v_perc_pix': 0.198,
     'n2v_patch_shape': (64, 64),
     'n2v_manipulator': 'mean',
     'n2v_neighborhood_radius': 5,
     'single_net_per_channel': True,
     'blurpool':True,
     'skip_skipone':True,
     'structN2Vmask': None,
     'probabilistic': False}
    ```
    

In order to reproduce the result using the plugin, we then follow these steps:

1. Download the [data](https://download.fht.org/jug/n2v/SEM.zip) and unzip it. Place the train and validation images in two different folders.
2. Start the plugin and point towards the train and validation folders in the `From disk`.
3. Confirm that your environment is properly set for GPU training by checking that the GPU indicator (top right) in the plugin displays a greenish GPU label.
4. Make sure to enter `YX` in `Axes`.
5. In `Training parameters`, set: <br>
`N epochs` = 20 <br>
`N steps` = 27 <br>
`Batch size` = 128 <br>
`Patch XY` = 64 <br>
6. Click on the gear button to open the `Expert settings` and set: <br>
`U-Net kernel size` = 3 <br>
`N2V2` = checked <br>
7. You can compare the configuration above to the rest of the `Expert settings` to confirm that the other default values are properly set.
8. Train!
9. Note that for the prediction, you will probably need to use tiling.

## 2D structN2V Convollaria

The example notebook can be found [here](https://github.com/juglab/n2v/blob/master/examples/2D/structN2V_2D_convallaria/01_training.ipynb).

```bash
config = N2VConfig(X, 
                   unet_kern_size=3, 
                   train_steps_per_epoch=500, 
                   train_epochs=10, 
                   train_loss='mse', 
                   batch_norm=True, 
                   train_batch_size=128, 
                   n2v_perc_pix=0.198, 
                   n2v_patch_shape=(64, 64), 
                   n2v_manipulator='uniform_withCP', 
                   n2v_neighborhood_radius=5, 
                   structN2Vmask = [[0,1,1,1,1,1,1,1,1,1,0]])
```

- Complete configuration
    
    ```bash
    {'means': ['549.53174'],
     'stds': ['105.084'],
     'n_dim': 2,
     'axes': 'YXC',
     'n_channel_in': 1,
     'n_channel_out': 1,
     'unet_residual': False,
     'unet_n_depth': 2,
     'unet_kern_size': 3,
     'unet_n_first': 32,
     'unet_last_activation': 'linear',
     'unet_input_shape': (None, None, 1),
     'train_loss': 'mse',
     'train_epochs': 10,
     'train_steps_per_epoch': 500,
     'train_learning_rate': 0.0004,
     'train_batch_size': 128,
     'train_tensorboard': True,
     'train_checkpoint': 'weights_best.h5',
     'train_reduce_lr': {'factor': 0.5, 'patience': 10},
     'batch_norm': True,
     'n2v_perc_pix': 0.198,
     'n2v_patch_shape': (64, 64),
     'n2v_manipulator': 'uniform_withCP',
     'n2v_neighborhood_radius': 5,
     'single_net_per_channel': True,
     'structN2Vmask': [[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]],
     'probabilistic': False}
    ```
    

1. Download the [data](https://download.fht.org/jug/n2v/flower.tif) and load it into napari.
2. Confirm that your environment is properly set for GPU training by checking that the GPU indicator (top right) in the plugin displays a greenish GPU label.
3. In `Training parameters`, set: <br>
`N epochs` = 10 <br>
`N steps` = 500 <br>
`Batch size` = 128 <br>
`Patch XY` = 64 <br>
4. Click on the gear button to open the `Expert settings` and set: <br>
`U-Net kernel size` = 3 <br>
`structN2Vmask` = 0,1,1,1,1,1,1,1,1,1,0 <br>
5. You can compare the configuration above to the rest of the `Expert settings` to confirm that the other default values are properly set.
6. Train!

## 3D example

The example notebook can be found [here](https://github.com/juglab/n2v/blob/master/examples/3D/01_training.ipynb).

```bash
config = N2VConfig(X, 
                   unet_kern_size=3, 
                   train_steps_per_epoch=4,
                   train_epochs=20, 
                   train_loss='mse', 
                   batch_norm=True, 
                   train_batch_size=4, 
                   n2v_perc_pix=0.198, 
                   n2v_patch_shape=(32, 64, 64), 
                   n2v_manipulator='uniform_withCP', 
                   n2v_neighborhood_radius=5)
```

- Full configuration
    
    ```bash
    {'means': ['37.057674'],
     'stds': ['5.3700876'],
     'n_dim': 3,
     'axes': 'ZYXC',
     'n_channel_in': 1,
     'n_channel_out': 1,
     'unet_residual': False,
     'unet_n_depth': 2,
     'unet_kern_size': 3,
     'unet_n_first': 32,
     'unet_last_activation': 'linear',
     'unet_input_shape': (None, None, None, 1),
     'train_loss': 'mse',
     'train_epochs': 20,
     'train_steps_per_epoch': 4,
     'train_learning_rate': 0.0004,
     'train_batch_size': 4,
     'train_tensorboard': True,
     'train_checkpoint': 'weights_best.h5',
     'train_reduce_lr': {'factor': 0.5, 'patience': 10},
     'batch_norm': True,
     'n2v_perc_pix': 0.198,
     'n2v_patch_shape': (32, 64, 64),
     'n2v_manipulator': 'uniform_withCP',
     'n2v_neighborhood_radius': 5,
     'single_net_per_channel': True,
     'structN2Vmask': None,
     'probabilistic': False}
    ```
    
1. In napari, go to `File / Open sample / napari-n2v / Download data (3D).`
2. Confirm that your environment is properly set for GPU training by checking that the GPU indicator (top right) in the plugin displays a greenish GPU label.
3. Check `Enable 3D`. 
4. In `Training parameters`, set: <br>
`N epochs` = 20 <br>
`N steps` = 4 <br>
`Batch size` = 4 <br>
`Patch XY` = 64 <br>
`Patch Z` = 32 <br>
5. Click on the gear button to open the `Expert settings` and set: <br>
`U-Net depth` = 2 <br>
`U-Net kernel size` = 3 <br>
6. You can compare the configuration above to the rest of the `Expert settings` to confirm that the other default values are properly set.
7. Train!
8. Note that for the prediction, you will probably need to use tiling.
