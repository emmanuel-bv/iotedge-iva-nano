#!/bin/bash

# Arguments
iotCentralScopeId=$1
iotCentralDeviceName=${2:-$(hostname)}
if [ -z "$iotCentralScopeId" ]; then
        echo "Missing  IoT Central Scope Id argument. Exiting."
        exit 1
fi

echo "1/5 - Verifying pre-requisites"
#Internet check
if ping -q -c 1 -W 1 8.8.8.8 >/dev/null; then
  echo "Please double check your internet connectivity. Exiting."
  exit 1
fi
#Sudo check
if [ "$EUID" -ne 0) ]; then
  echo "Please run this script as root. Exiting."
  exit 1
fi
#Nvidia docker check
if [ -z $(nvidia-docker --version) ]; then
  echo "Please double check that nvidia-docker is installed properly (which is the case on the default image provided by NVIDIA). Exiting."
  exit 1
fi
#CUDA version check
if [ "CUDA_VER" -ne 10.0 ]; then
  echo "This script only works with CUDA VERSION 10.0. Exiting."
  exit 1
fi
#TODO: Verify that all tools are installed: curl, sed. If not install them.

echo "2/5 - Installing IoT Edge"
echo ""
curl https://packages.microsoft.com/config/ubuntu/18.04/multiarch/prod.list > ./microsoft-prod.list
sudo cp ./microsoft-prod.list /etc/apt/sources.list.d/curl https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > microsoft.gpg
sudo cp ./microsoft.gpg /etc/apt/trusted.gpg.d/
curl https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > microsoft.gpg
sudo cp ./microsoft.gpg /etc/apt/trusted.gpg.d/
sudo apt-get update
sudo apt-get install iotedge

echo "3/5 - Downloading configuration and sample files"
cd /var
#TODO -- check permissions of downloaded file
wget -O streams.tar.gz --no-check-certificate "https://onedrive.live.com/download?cid=0C0A4A69A0CDCB4C&resid=C0A4A69A0CDCB4C%21585942&authkey=AK97kEm8xWwybjo"
tar -xzvf streams.tar.gz

echo "4/5 - Configuring IoT Edge to connect to your IoT Central instance (Scope Id: " $iotCentralScopeId ", DeviceId = " $iotCentralDeviceName ")"
echo ""
sudo rm /etc/iotedge/config.yaml
sudo mv /var/iotedge/config.yaml /etc/iotedge/config.yaml
emptyIotEdgeConfigScopeIdLine="  device_connection_string:"
emptyIoTEdgeConfigDeviceNameLine="TBD:" #TODO
completeIoTEdgeConfigScopeIdLine="${emptyIoTEdgeConfigScopeIdLine} ${iotCentralScopeId}"
completeIoTEdgeConfigDeviceNameLine="${emptyIoTEdgeConfigDeviceNameLine} ${iotCentralDeviceName}"
sudo sed -i 's/${emptyIoTEdgeConfigScopeIdLine}/${completeIoTEdgeConfigScopeIdLine}' /etc/iotedge/config.yaml
sudo sed -i 's/${emptyIoTEdgeConfigDeviceNameLine}/${completeIoTEdgeConfigDeviceNameLine}' /etc/iotedge/config.yaml
sudo systemctl restart iotedge


echo "5/5 - Caching IoT Edge modules"
echo ""
docker pull mcr.microsoft.com/azureiotedge-agent:1.0
docker pull mcr.microsoft.com/azureiotedge-hub:1.0
docker pull marketplace.azurecr.io/nvidia/deepstream-iot-l4t:latest
#TODO: ADD IoT Central Bridge module and Image Tagging Module

echo "6/6 - Configuring IoT Edge modules"
#TODO - Copy Device Info config file
#TODO - Copy Deepstream cofig template file

echo "All steps completed. Your current display number is " $DISPLAY ". Please use the corresponding 'Edge IVA - Display number' device template in IoT Central."
echo "Please reboot to finalize the setup and connect to IoT Central to manage your device".