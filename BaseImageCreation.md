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

5. Comment out all TensorFlow or PyTorch install related steps and Save
6. Install the USB-Device-Mode Jupyter notebook by running this script:

```bash
./install.sh
```
