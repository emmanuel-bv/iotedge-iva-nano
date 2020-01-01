# Instructions

## Prerequisites

- **Hardware**: You need a [NVIDIA Jetson Nano device](https://developer.nvidia.com/embedded/buy/jetson-nano-devkit) ideally with a [5V-4A barrel jack power supply like this one](https://www.adafruit.com/product/1466), which requires a jumper cable (such as [these ones](https://www.amazon.com/120pcs-Multicolor-Jumper-Arduino-Raspberry/dp/B01BAXKDN4/ref=asc_df_B01BAXKDN4/?tag=hyprod-20&linkCode=df0&hvadid=198075247191&hvpos=1o1&hvnetw=g&hvrand=12715964868364075974&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=9033288&hvtargid=pla-317965496827&psc=1)) on pins J48. See the [Power Guide section of the Jetson Nano Developer Kit](https://developer.nvidia.com/embedded/dlc/jetson-nano-dev-kit-user-guide) for more details. Alternatively, a 5V-2.5A Micro-USB power supply will work without a jumper cable but may limit the performance of your Deepstream application. In all cases, please make sure to use the default `Max` power source mode (e.g. 10W). To visualize the video feeds, you'll need an HDMI monitor and cable connected to your NVIDIA Jetson Nano.
- **Install Jetson Nano base image**: [install its base image](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit). It is based on Ubuntu 18.04 and already includes NVIDIA drivers version > 418, CUDA and Nvidia-Docker.
- **Azure Subscription**: you need an Azure subscription to deploy an IoT Central application and manage your devices remotely.

## Temporary get started to use Custom Vision models

- Set up Scott's demo
- Copy on the Nano device the following files to use with an 2-classes object detection AI model built with Custom Vision  https://1drv.ms/u/s!AkzLzaBpSgoMo-x6jQitjjlAyjXytQ?e=YRYcL3
- Build the updated IoTCentralBridge module from this repo
- Update the IoTCentral Bridge module to craft a config file that maps to the one copied above (main config file and config_infer_primary)
- Use the updated deployment manifest from this repo and migrate the device to his version in IoT Central

Note: the current libnvdsinfer_custom_impl_Yolo.so only works with 2 classes but I'll provide one that works with n-classes defined in config_infer_primary file soon.

## Get Started

//TODO: detail these steps

- Deploy your Azure IoT Central solution based on this template.
- Copy your ScopeId.
- Copy the `setup.sh` script to your NVIDIA Jetson Nano device
- Run this one-time setup script on your NVIDIA Jetson Nano device with your ScopeId as parameter:

```bash
./setup.sh yourScopeId
```
