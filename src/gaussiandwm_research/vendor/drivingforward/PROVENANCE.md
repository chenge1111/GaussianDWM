Network sources from https://github.com/fangzhou2000/DrivingForward at
5d0a2c7d6a6358f55318faac80e351cf75dff032, licensed MIT, copyright Qijian Tian.

Included: network/depth_network.py, network/volumetric_fusionnet.py,
network/blocks.py, models/gaussian/gaussian_network.py and extractor.py.
Imports use local modern-PyTorch compatibility functions instead of
external PackNet/PyTorch3D dependencies. compat.py is new extension code.
The official depth/GS checkpoint naming is retained. Dataset, original training
wrapper and original CUDA rasterizer are not copied; this project uses gsplat.
