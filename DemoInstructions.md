# Visual Anomaly Detection over multiple cameras with NVIDIA Jetson Nano devices workshop

These instructions provide the fastest path to demo a real-time video analytics application processing multiple video streams with a custom AI model on a $100 device, and how to operate it remotely.

## Prerequisites

- **Hardware**: You need a [NVIDIA Jetson Nano device](https://developer.nvidia.com/embedded/buy/jetson-nano-devkit) ideally with a [5V-4A barrel jack power supply like this one](https://www.adafruit.com/product/1466), which requires a jumper cable (such as [these ones](https://www.amazon.com/120pcs-Multicolor-Jumper-Arduino-Raspberry/dp/B01BAXKDN4/ref=asc_df_B01BAXKDN4/?tag=hyprod-20&linkCode=df0&hvadid=198075247191&hvpos=1o1&hvnetw=g&hvrand=12715964868364075974&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=9033288&hvtargid=pla-317965496827&psc=1)) on pins J48. See the [Power Guide section of the Jetson Nano Developer Kit](https://developer.nvidia.com/embedded/dlc/jetson-nano-dev-kit-user-guide) for more details. Alternatively, a 5V-2.5A Micro-USB power supply will work without a jumper cable but may limit the performance of your Deepstream application. In all cases, please make sure to use the default `Max` power source mode (e.g. 10W). To visualize the video feeds, you'll need an HDMI monitor and cable connected to your NVIDIA Jetson Nano.

![Jetson Nano](./assets/JetsonNano.png "NVIDIA Jetson Nano device used to run Deepstream with IoT Edge")



- **A laptop**: You need a laptop to connect to your Jetson Nano device and see its results with a browser and an ssh client.
- **VLC**: To visualize the output of the Jetson Nano without HDMI screen (there is only one per table), we'll use VLC from your laptop to view a RTSP video stream of the processed videos. [Install VLC](https://www.videolan.org/vlc/index.html) if you dont have it yet.
- **Have an Azure subscription**: You need an Azure subscription to create an Azure IoT Central  application.
- **Have a phone with IP Camera Lite app**: To view & process a live video stream, you can use your phone with the IP Camera Lite app ([iOS](https://apps.apple.com/us/app/ip-camera-lite/id1013455241), [Android](https://play.google.com/store/apps/details?id=com.shenyaocn.android.WebCam&hl=en_US)) as an IP camera.

## Setting up a new IoT Central app

Let's create a new IoT Central app to which we will connect our Jetson Nano later on.

### Create a new IoT Central app

We'll start from a pre-built template of IoT Central, which already includes an application to see and command a video analytics solution running on the NVIDIA Jetson Nano.

- From your browser, go to: https://apps.azureiotcentral.com/build/new/af1fe1b4-d92e-45fc-9d1f-ea4decdc961d
- Give a name and URL to your application
- Select your Azure subscription (you can opt-in for a 7 day free trial)
- Select your location
- Click on `Create`

### Create an IoT Edge device from your IoT Central app

We'll create a new IoT Edge device in your IoT Central application that will enable to the NVIDIA Jetson Nano to connect to IoT Central.

- Go to the `Devices` tab
- Select the `NVIDIA Jetson Nano DCM` device template
- Click on `New`
- Give a name to your device by editing the `Device ID` and the `Device name` fields (let's use the same name for both of these fields in this workshop)
- Click on `Create`
- Click on your new device
- Click on the `Connect` button in the top right corner
- Copy your `ID Scope` value, `Device ID` value and `Primary key` value and save them for later.

## Setting up your device

We'll start from a blank Jetson installation (Jetpack v4.3), copy a few files locally that are needed for the application such as video files to simulate RTSP cameras and deepstream configuration files, and install IoT Edge.

1. **Flash your Jetson Nano SD Card**: download and flash either this [JetPack version 4.3 image](https://developer.nvidia.com/embedded/jetpack) if you have an HDMI screen or the [image from this NVIDIA course](https://courses.nvidia.com/courses/course-v1:DLI+C-IV-02+V1/info) otherwise. The image from the course is also based on JetPack 4.3 but includes an USB Device Mode to use the Jetson Nano without HDMI screen. For the rest of this tutorial will assume that you use the device in USB Device Mode. In any cases, you can use [BalenaEtcher](https://www.balena.io/etcher/) tool to flash your SD card. Both of these images are based on Ubuntu 18.04 and already includes NVIDIA drivers version, CUDA and Nvidia-Docker. 

```bash
head -n 1 /etc/nv_tegra_release
```

2. **Connect your Jetson Nano to your developer's machine with the USB Device Mode**: we'll do that by plugging a micro-USB cable from your Jetson Nano to your developer's machine and using the USB Device Mode provided in NVIDIA's course base image. With this mode, you do not need to hook up a monitor directly to your Jetson Nano. Instead, boot your device and wait for 30 seconds then open yoru favorite browser, go to [http://192.168.55.1:8888](http://192.168.55.1:8888) and enter the password `dlinano` to get access to a command line on your Jetson Nano.

![Jupyter Notebook](./assets/JupyterNotebook.png "Jetson Nano controlled by a Jupyter Notebook via the USB Device Mode")

3. **Connect your Jetson Nano to the internet**: Either use an ethernet connection, in which case you can skip this section or connect your device to WiFi using the above USB Device Mode terminal:

    To connect your Jetson to a WiFi network from a terminal, follow these steps

    1. Re-scan available WiFi networks

        ```bash
        nmcli device wifi rescan
        ```

    2. List available WiFi networks, and find the ``ssid_name`` of your network.

        ```bash
        nmcli device wifi list
        ```

    3. Connect to a selected WiFi network

        ```bash
        nmcli device wifi connect <ssid_name> password <password>
        ```

4. **Connect your Jetson Nano to an SSH client**: The USB Device Mode terminal is limited because it does not support copy/paste. So to make it easier to go through the steps of this sample, it is recommended to open an SSH connection with your favorite SSH Client.

    1. Find your IP address using the USB Device Mode terminal

        ```bash
        ifconfig
        ```

    2. Make sure that your laptop is on the same network as yoru Jetson Nano device and open an SSH connection on your Jetson Device (password = `dlinano`):

        ```bash
        ssh dlinano@your-ip-address
        ```

5. **Download setup files**:
    1. On your Jetson Nano create a folder name `data` at the root:

        ```bash
        sudo mkdir /data
        ```

    2. Download and extra setup files in the `data` directory:

        ```bash
        cd /data
        sudo wget -O setup.tar.bz2 --no-check-certificate "https://onedrive.live.com/download?cid=0C0A4A69A0CDCB4C&resid=0C0A4A69A0CDCB4C%21588477&authkey=AIWlT_q7sPbcfS4"
        sudo tar -xjvf setup.tar.bz2
        ```

    3. Make the folder accessible from a normal user account:
    
        ```bash
        sudo chmod -R 777 /data
        ```

4. **Install IoT Edge:** Follow these instructions copied [from here](https://docs.microsoft.com/en-us/azure/iot-edge/how-to-install-iot-edge-linux):

    ```bash
    curl https://packages.microsoft.com/config/ubuntu/18.04/multiarch/prod.list > ./microsoft-prod.list
    sudo cp ./microsoft-prod.list /etc/apt/sources.list.d/
    curl https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > microsoft.gpg
    sudo cp ./microsoft.gpg /etc/apt/trusted.gpg.d/
    sudo apt-get update
    sudo apt-get install iotedge
    ```

5. **Connect your device to your IoT Central application:** edit IoT Edge configuration file:

    1. Use your favorite text editor to edit IoT Edge configuration file:

    ```bash
    sudo nano /etc/iotedge/config.yaml
    ```

    2. Comment out the "Manual provisioning configuration" section so it looks like this:

    ```bash
    # Manual provisioning configuration
    #provisioning:
    #  source: "manual"
    #  device_connection_string: ""
    ```

    3. Uncomment the "DPS symmetric keyi provisioning configuration" and add your IoT Central app's scope id, registration_id which is your device Id and its primary symmetric key:

        ```bash
        # DPS symmetric key provisioning configuration
        provisioning:
        source: "dps"
        global_endpoint: "https://global.azure-devices-provisioning.net"
        scope_id: "<ID Scope>"
        attestation:
            method: "symmetric_key"
            registration_id: "<Device ID>"
            symmetric_key: "<Primary Key>"
        ```

    4. Save and exit your editor (Ctrl+O, Ctrl+X)

    5. Now Restart the Azure IoT Edge runtime with the following command:

    ```bash
    sudo systemctl restart iotedge
    ```

    6. After a few moments the Edge runtime restarts and establishes a connection with your IoT Central application.

Note that the first time that you connect your IoT Edge device to IoT Central, it will take time to start the entire solution since it first needs to download ~2Gbs of IoT Edge modules.

## Running the solution

1. Connect to your IoT Central application
2. Go to `Devices` and find the device number corresponding to your Jetson Nano (You should have received this number from the proctors and will range from 01-45, for example if you received 01 then your device will be jetson-nano-01)
3. Click on this device and go to the `Dashboard` tab
4. Verify that active telemetry is being sent by the device to IoT Central
5. Copy the `RTSP Video URL` from the `Device` tab
6. Open VLC and go to `Media` > `Open Network Stream` and paste the `RTSP Video URL` copied above as the network URL and click `Play`

At this point, you should see 4 video streams being processed to detect cars and people with a Resnet 10 AI model.

![4 video streams processed real-time](./assets/4VideosProcessedRTSP.png "8 video streams processed in real-time by a Jetson Nano with Deepstream and IoT Edge")

## Operating the solution

To demonstrate how to remotely manage this solution, we'll send a command to the device to change its input cameras. We'll use your phone as an RTSP camera as a new input camera.

![IoT Central](./assets/IoTCentral.png "IoT Central UI to remotely manage NVIDIA Jetson devices")

### Changing input cameras

Let's first verify that your phone works as an RTSP camera properly:

- Open the the IP Camera Lite
- Go to Settings and remove the User and Password on the RTSP feed
- Click on `Turn on IP Camera Server`


Let's just verify that the camera is functional. With VLC:

- Go to `Media` > `Open Network Stream`
- Paste the following `RTSP Video URL`:  `rtsp://your-phone-ip-address:8554/live`
- Click `Play` and verify that phone's camera is properly displaying.

Let's now update your Jetson Nano to use your phone's camera. In IoT Central:

- Go to the `Manage` tab
- Unselect the `Demo Mode`, which uses several hardcoded video files as input of car traffic
- Update the `Video Stream 1` property:
    - In the `cameraId`, name your camera, for instance `My Phone`
    - In the `videoStreamUrl`, enter the RTSP stream of this camera: `rtsp://your-phone-ip-address:8554/live`
- Keep the default AI model of DeepStream by keeping the value `DeepStream ResNet 10` as the `AI model type`.
- Keep the default `Primary Detection Class` as `person`
- Hit `Save`

This sends a command to the device to update its DeepStream configuration file with these new properties and to restart DeepStream. If you were still streaming the output of the DeepStream application, this streem will be taken down as DeepStream will restart.

Within a minute, DeepStream should restart. You can observe its status in IoT Central via the `Modules` tab. Once `deepstream` module is back to `Running`, copy again the `RTSP Video Url` field from the `Device` tab and give it to VLC (`Media` > `Open Network Stream` > paste the `RTSP Video URL` > `Play`).

You should now detect people from your phone's camera. The count of `Person` in the `dashboard` tab in IoT Central should go up. We've just remotely updated the configuration of this intelligent video analytics solution!

## Use an AI model to detect custom visual anomalies

A soda can manufaturer wants to improve the efficienty of its plant. He would like to be able to take soda cans that fell down on his production lines to avoid slow downs.

We'll use cameras to monitor each of the lines and we'll collect images and build a custom AI model to detects cans that are up or down. We'll then deploy this custom AI model to DeepStream via IoT Central. To do a quick Proof Of Concept, we'll use the [Custom Vision service](https://www.customvision.ai/), a no-code computer vision AI model builder.

A model has already been built with [Custom Vision ](https://www.customvision.ai/) using [this set of images](https://1drv.ms/u/s!AEzLzaBpSgoMo-1l). You can find [a pre-built Custom Vision model here](https://onedrive.live.com/download?0C0A4A69A0CDCB4C&resid=0C0A4A69A0CDCB4C%21587636&authkey=AOCf3YsqcZM_3WM).

We'll now deploy this custom vision model to the Jetson Nano using IoT Central and update the simulated cameras to map to the ones in soda can factory. In IoT Central:

- Go to the `Manage` tab (beware of the sorting o f the fields)
- Make sure the `Demo Mode` is unchecked
- Update the first three `Video Stream Input` to the following values:
    - `Video Stream Input 1` > `CameraId` = `Cam01`
    - `Video Stream Input 1` > `videoStreamUrl` = `file:///data/misc/storage/sampleStreams/cam-cans-00.mp4`
    - `Video Stream Input 2` > `CameraId` = `Cam02`
    - `Video Stream Input 2` > `videoStreamUrl` = `file:///data/misc/storage/sampleStreams/cam-cans-01.mp4`
    - `Video Stream Input 3` > `CameraId` = `Cam03`
    - `Video Stream Input 3` > `videoStreamUrl` = `file:///data/misc/storage/sampleStreams/cam-cans-02.mp4`
- Select `Custom Vision` as the `AI model Type`
- Paste the URI of your custom vision model in the `Custom Vision Model Url`, for instance `https://onedrive.live.com/download?0C0A4A69A0CDCB4C&resid=0C0A4A69A0CDCB4C%21587636&authkey=AOCf3YsqcZM_3WM` for the pre-built one.
- Update the detection classes:
    -  `Primary Detection Class` = `Up`
    - `Secondary Detection Class` = `Down`
- Hit `Save`

After a few moments, the `deepstream` module should restart. Once it is in `Running` state again, look at the output RTSP stream via VLC (`Media` > `Open Network Stream` > paste the `RTSP Video URL` that you got from the IoT Central's `Device` tab > `Play`).

We are now visualizing the processing of 3 real time (e.g. 30fps 1080p) video feeds with a custom vision AI models that we built in minutes to detect visual anomalies!

![Custom Vision](./assets/sodaCans.png "3 soda cans manufacturing lines are bieing monitored with a custom AI model built with Custom Vision")
