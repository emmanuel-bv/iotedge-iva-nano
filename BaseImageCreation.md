# Instructions to create your base Jetson Nano image from scratch

1. Flash on a 32Gb+ SD card, the Jetpack version that you need from NVIDIA's website (JetPack 4.3 is the latest one)
2. Insert the SD card in yoru Jetson Nano and go through the Out of Box Experience
3. In a terminal, run the following commands:

```bash
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install curl nano
```

4. 

```bash
git clone -b jetpack_4.3 https://github.com/NVIDIA-AI-IOT/jetcard
cd jetcard
nano ./install.sh
```

5. Comment out all TensorFlow, PyTorch and TF-models install related steps and Save
6. Install the USB-Device-Mode Jupyter notebook by running this script:

```bash
./install.sh
```

7. To pre-cache container images, download the following images:

```bash
sudo docker pull mcr.microsoft.com/azureiotedge-agent:1.0
sudo docker pull mcr.microsoft.com/azureiotedge-hub:1.0
sudo docker pull marketplace.azurecr.io/nvidia/deepstream-iot2-l4t:latest
sudo docker pull ebertrams/iotcentralbridge:0.1.0-arm64v8
```

8. Set docker to rotate the logs:

    - Edit this docker configuration

    ```bash
    sudo nano /etc/docker/daemon.json
    ```

    - Add the following lines:

    ```json
    {
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "10m",
        "max-file": "5"
        }
    }
    ```

8. Remove the SD card from your Jetson Nano
9. Insert the SD card into a Linux host computer
10. Print the partition table to read sectors with `fdisk` (assuming that sdb maps to yoru SD Card)

```bash
sudo fdisk /dev/sdb
```

11. Not the end sector of the end parition
12. Add 1
13. Divide by 2048 to convert to megabytes
14. Round up to the nearest whole number to get the last block to copy on the SD card

15. Create an image out of the SD card (without the unused space):

```
sudo dd bs=1M if=/dev/sdb of=jetpack_4.3_usb_mode.img status=progress count=<your block count>
```

16. Compress your image

```
zip jetpack_4.3_usb_mode_img.zip jetpack_4.3_usb_mode.img
```

17. Flash your next SD card with this image using [BalenaEtcher](https://www.balena.io/etcher/)
