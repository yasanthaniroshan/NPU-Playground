# NPU Playground

This repository contains experiments and examples for working with NPU in Luckfox Pico Mini B (rockchip rv1103) using the RKNN Toolkit.

## Table of Contents
- [Introduction](#introduction)
- [Setup Environment](#setup-environment)
- [Convert Models](#convert-models)
- [Run Inference](#run-inference)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Introduction

The NPU Playground is designed to help developers explore the capabilities of the Neural Processing Unit (NPU) on the Luckfox Pico Mini B. It guides users through setting up the environment, converting models to RKNN format, and running inference on the device.

## Setup Environment

> [!NOTE]
> You need linux based OS to setup the environment.

To set up the environment for NPU development, follow these steps:
1. Install the RKNN Toolkit by following the instructions in the [RKNN Toolkit Documentation](https://wiki.luckfox.com/Luckfox-Pico-Plus-Mini/RKNN)
    - Download RKNN-Toolkit2
    ```bash
    git clone https://github.com/airockchip/rknn-toolkit2
    cd rknn-toolkit2
    ```
    - Install dependencies
    ```bash
    sudo apt-get update
    sudo apt-get install python3 python3-dev python3-pip
    sudo apt-get install libxslt1-dev zlib1g zlib1g-dev libglib2.0-0 libsm6 libgl1 mesa-utils libglib2.0-0t64 libprotobuf-dev gcc
    ```

    - Install RKNN-Toolkit2
    ```bash
    pip3 install -r rknn-toolkit2/packages/requirements_cpxx-1.6.0.txt
    pip3 install rknn-toolkit2/packages/rknn_toolkit2-x.x.x+xxxxxxxx-cpxx-cpxx-linux_x86_64.whl
    ```

    Here xx is the python version you are using, e.g., cp36 for python 3.6.

    - Verify installation
    ```bash
    python3 -c "from rknn.api import RKNN; print('RKNN Toolkit installed successfully')"
    ```
2. Install additional dependencies for model conversion and inference as needed.
    ```bash
    pip install -r requirements.txt
    ```
3. Connect device and see the libc types
    ```bash
    ls /lib
    ```
4. Check for the RKNN NPU driver
    ```bash
    ls /dev | grep rknn
    ```
5. If the rknn device is not found, install the NPU driver by following the instructions in the [NPU Driver Installation Guide](https://wiki.luckfox.com/Luckfox-Pico-Plus-Mini/RKNN).

6. Copy the shared libraries to the device (follow the guidelines in the [SDK Quick Start](SDK%20V2.3.2.pdf) | (3.4 Install RKNPU2 Environment on the Board))
    ```bash
    sudo cp /usr/lib/aarch64-linux-gnu/librknn_api.so* /usr/lib/
    sudo ldconfig
    ```


## Convert Models
To convert models to RKNN format, use the RKNN Toolkit. Here is an example of converting a TensorFlow model in [main.py](./../main.py)

## Run Inference
To run inference on the NPU, use the RKNN Toolkit's inference API. An example can be found in [Model Inference](./../Model-Inference/)

## Examples
- Not included yet.

## Troubleshooting
If you encounter issues during setup or model conversion, refer to the following troubleshooting tips:
- Ensure all dependencies are installed correctly.
- Verify that the NPU driver is installed and the device is recognized.
- Check the RKNN Toolkit documentation for specific error messages.

## License
This project is licensed under the MIT License. See the [LICENSE](./../LICENSE) file