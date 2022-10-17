# NanoFlowNet: Real-time Dense Optical Flow on a Nano Quadcopter

Nano quadcopters are small, agile, and cheap platforms that are well suited for deployment in narrow, cluttered environments. Due to their limited payload, these vehicles are highly constrained in processing power, rendering conventional vision-based methods for safe and autonomous navigation incompatible. Recent machine learning developments promise high-performance perception at low latency, while dedicated edge computing hardware has the potential to augment the processing capabilities of these limited devices. In this work, we present NanoFlowNet, a lightweight convolutional neural network for real-time dense optical flow estimation on edge computing hardware. We draw inspiration from recent advances in semantic segmentation for the design of this network. Additionally, we guide the learning of optical flow using motion boundary ground truth data, which improves performance with no impact on latency. Validation results on the MPI-Sintel dataset show the high performance of the proposed network given its constrained architecture. Additionally, we successfully demonstrate the capabilities of NanoFlowNet by deploying it on the ultra-low power GAP8 microprocessor and by applying it to vision-based obstacle avoidance on board a Bitcraze Crazyflie, a 34 g nano quadcopter. 

## Installation

The easiest way to set up is to install all requirements into a docker environment:

`docker run -v `<i>< path to FlyingChairs2 ></i>`:/workspace/FlyingChairs2 -v `<i>< path to flow datasets dir ></i>`:/workspace/flowData -v `<i>< path to this repo ></i>`:/workspace/nanoflownet --gpus all -it tensorflow/tensorflow:2.8.0-gpu`

Inside the created docker container:

`pip install opencv-python-headless==4.5.5.64`

`pip install tensorflow_model_optimization==0.7.2`

`pip install tqdm==4.64.0`

`pip install tensorflow_addons==0.16.1`

`pip install wandb==0.12.14`

`pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-tf-plugin-cuda110==1.12.0`

in another terminal find the docker container id with `docker ps -l`

and commit the changes

`docker commit [container_id] nanoflownet`

This concludes the set-up. The correct container can be now opened (without re-installing the pip requirements) by replacing `tensorflow/tensorflow:2.8.0-gpu` with `nanoflownet`:

`docker run -v `<i>< path to FlyingChairs2 ></i>`:/workspace/FlyingChairs2 -v `<i>< path to flow datasets dir ></i>`:/workspace/flowData -v `<i>< path to this repo ></i>`:/workspace/nanoflownet --gpus all -it nanoflownet`

## Links
[![arXiv](https://img.shields.io/badge/arXiv-2209.06918-b31b1b.svg)](https://arxiv.org/abs/2209.06918)