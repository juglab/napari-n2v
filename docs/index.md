# napari-n2v

`napari-n2v` brings [Noise2Void](https://github.com/juglab/n2v) to the fantastic world of napari. N2V is a sef-supervised denoising algorithm allowing 
removing pixel-independent noise. It also includes an extension, structN2V, aimed at removing structured noise.

This set of plugins can train, retrain and predict on images from napari or from the disk. It conveniently allows saving 
the models for later use and is compatible with [Bioimage.io](https://bioimage.io/#/). 


<img src="https://raw.githubusercontent.com/juglab/napari-n2v/master/docs/images/training.gif" width="800" />

# Documentation

1. [Installation](installation.md)
2. [Documentation](documentation.md)
3. [Examples](examples.md)
4. [Troubleshooting](faq.md)

# Report issues and errors

Help us improve the plugin by submitting [issues to the Github repository](https://github.com/juglab/napari-n2v/issues) 
or tagging @jdeschamps on [image.sc](https://forum.image.sc/). 

# Citation

### N2V

Alexander Krull, Tim-Oliver Buchholz, and Florian Jug. "[Noise2void-learning denoising from single noisy images.](https://ieeexplore.ieee.org/document/8954066)" 
*Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*. 2019.

### structN2V

Coleman Broaddus, et al. "[Removing structured noise with self-supervised blind-spot networks.](https://ieeexplore.ieee.org/document/9098336)" *2020 IEEE 17th 
International Symposium on Biomedical Imaging (ISBI)*. IEEE, 2020.

### N2V2

Eva Höck, Tim-Oliver Buchholz, et al. "[N2V2 - Fixing Noise2Void Checkerboard Artifacts with Modified Sampling Strategies and a Tweaked Network Architecture](https://openreview.net/forum?id=IZfQYb4lHVq)", (2022).


# Support 

This plugin was developed thanks to the support of the Silicon Valley Community Foundation (SCVF) and the 
Chan-Zuckerberg Initiative (CZI) with the napari Plugin Accelerator grant _2021-240383_.