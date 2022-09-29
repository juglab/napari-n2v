# napari-n2v

[![License](https://img.shields.io/pypi/l/napari-n2v.svg?color=green)](https://github.com/githubuser/napari-n2v/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-n2v.svg?color=green)](https://pypi.org/project/napari-n2v)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-n2v.svg?color=green)](https://python.org)
[![tests](https://github.com/juglab/napari-n2v/workflows/tests/badge.svg)](https://github.com/githubuser/napari-n2v/actions)
[![codecov](https://codecov.io/gh/juglab/napari-n2v/branch/main/graph/badge.svg)](https://codecov.io/gh/githubuser/napari-n2v)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-n2v)](https://napari-hub.org/plugins/napari-n2v)

A self-supervised denoising algorithm now usable by all in napari.

![Training](docs/images/training.gif)

----------------------------------

## Installation
<!---
You can install `napari-n2v` via [pip]:

    pip install napari-n2v

Or through the [napari-hub](https://napari.org/stable/plugins/find_and_install_plugin.html).
-->
Check out the documentation for installation instructions. (soon on PyPi and napari hub)


## Quick demo

![Demo prediction](docs/images/demo.gif)

You can try the quick demo by loading the "N2V Demo prediction" in plugins, and starting the prediction directly.
## Documentation

Documentation is available on the [project website](https://juglab.github.io/napari-n2v/).


## Contributing and feedback

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request. You can also 
help us improve by [filing an issue] along with a detailed description or contact us
through the [image.sc](https://forum.image.sc/) forum (tag @jdeschamps).


## Cite us

### N2V

Krull, Alexander, Tim-Oliver Buchholz, and Florian Jug. "Noise2void-learning denoising from single noisy images." 
*Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*. 2019.

### structN2V

Broaddus, Coleman, et al. "Removing structured noise with self-supervised blind-spot networks." *2020 IEEE 17th 
International Symposium on Biomedical Imaging (ISBI)*. IEEE, 2020.


## Acknowledgements

This plugin was developed thanks to the support of the Silicon Valley Community Foundation (SCVF) and the 
Chan-Zuckerberg Initiative (CZI) with the napari Plugin Accelerator grant _2021-240383_.


Distributed under the terms of the [BSD-3] license,
"napari-n2v" is a free and open source software.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[filing an issue]: https://github.com/githubuser/napari-n2v/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
