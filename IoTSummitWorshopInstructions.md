# Visual Anomaly Detection over multiple cameras with NVIDIA Jetson Nano devices workshop

In this workshop, you'll discover how to build a solution that can process up several real-time video streams with an AI model on a $100 device, how to remotely operate your device, and demonstrate how you can deploy custom AI models to it.

With this solution, you can transform cameras into sensors to know when there is an available parking spot, a missing product on a retail store shelf, an anomaly on a solar panel, a worker approaching a hazardous zone., etc.

We'll build this solution using [NVIDIA Deepstream](https://developer.nvidia.com/deepstream-sdk) on a [NVIDIA Jetson Nano device](https://developer.nvidia.com/embedded/buy/jetson-nano-devkit) connected to Azure via [Azure IoT Edge](https://azure.microsoft.com/en-us/services/iot-edge/). Deepstream is an highly-optimized video processing pipeline, capable of running deep neural networks. It is a must-have tool whenever you have complex video analytics requirements like real-time object detection or when employing cascading AI models. IoT Edge gives you the possibility to run this pipeline next to your cameras, where the video data is being generated, thus lowering your bandwitch costs and enabling scenarios with poor internet connectivity or privacy concerns.

We'll operate this solution with an aesthetic UI provided by [IoT Central](https://azure.microsoft.com/en-us/services/iot-central/) and customize the objects detected in video streams using [Custom Vision](https://www.customvision.ai/), a service that automatically generates computer vision AI models from pictures.

## Prerequisites

- **Hardware**: You need a [NVIDIA Jetson Nano device](https://developer.nvidia.com/embedded/buy/jetson-nano-devkit) ideally with a [5V-4A barrel jack power supply like this one](https://www.adafruit.com/product/1466), which requires a jumper cable (such as [these ones](https://www.amazon.com/120pcs-Multicolor-Jumper-Arduino-Raspberry/dp/B01BAXKDN4/ref=asc_df_B01BAXKDN4/?tag=hyprod-20&linkCode=df0&hvadid=198075247191&hvpos=1o1&hvnetw=g&hvrand=12715964868364075974&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=9033288&hvtargid=pla-317965496827&psc=1)) on pins J48. See the [Power Guide section of the Jetson Nano Developer Kit](https://developer.nvidia.com/embedded/dlc/jetson-nano-dev-kit-user-guide) for more details. Alternatively, a 5V-2.5A Micro-USB power supply will work without a jumper cable but may limit the performance of your Deepstream application. In all cases, please make sure to use the default `Max` power source mode (e.g. 10W). To visualize the video feeds, you'll need an HDMI monitor and cable connected to your NVIDIA Jetson Nano.

![Jetson Nano](./assets/JetsonNano.png "NVIDIA Jetson Nano device used to run Deepstream with IoT Edge")

- **Connect your Jetson Nano to your developer's machine with the USB Device Mode**: we'll do that by plugging a micro-USB cable from your Jetson Nano to your developer's machine and using the USB Device Mode provided in NVIDIA's course base image. With this mode, you do not need to hook up a monitor directly to your Jetson Nano. Instead, boot your device and wait for 30 seconds then open yoru favorite browser, go to [http://192.168.55.1:8888](http://192.168.55.1:8888) and enter the password `dlinano` to get access to a command line on your Jetson Nano.

![Jupyter Notebook](./assets/JupyterNotebook.png "Jetson Nano controlled by a Jupyter Notebook via the USB Device Mode")

- **Connect your Jetson Nano to an SSH client**: The USB Device Mode terminal is limited because it does not support copy/paste. So to make it easier to go through the steps of this sample, it is recommended to open an SSH connection with your favorite SSH Client. 

    1. Find your IP address using the USB Device Mode terminal

        ```bash
        ifconfig
        ```

    2. Make sure that your laptop is on the same network as yoru Jetson Nano device and open an SSH connection on your Jetson Device (password = `dlinano`):

        ```bash
        ssh dlinano@your-ip-address
        ```

- **Install IoT Edge**: See the [Azure IoT Edge installation instructions](https://docs.microsoft.com/en-us/azure/iot-edge/how-to-install-iot-edge-linux) for Ubuntu Server 18.04. Skip the Install Container Runtime section since we will be using nvidia-docker, which is already installed. Connect your device to your IoT Hub using the manual provisioning option. See this [quickstart](https://docs.microsoft.com/en-us/azure/iot-edge/quickstart-linux) if you don't yet have an Azure IoT Hub.
- **Install VS Code and its the IoT Edge extension on your developer's machine**: On your developer's machine, get [VS Code](https://code.visualstudio.com/) and its [IoT Edge extension](https://marketplace.visualstudio.com/items?itemName=vsciot-vscode.azure-iot-tools#overview). [Configure this extension with your IoT Hub](https://docs.microsoft.com/en-us/azure/iot-edge/how-to-deploy-modules-vscode#sign-in-to-access-your-iot-hub).
- **Install VLC to view RTSP video streams**: On your developer's machine, [install VLC](https://www.videolan.org/vlc/index.html).

The next sections walks you step-by-step to deploy Deepstream on an IoT Edge device and update its configuration. It explains concepts along the way. If all you want is to see the 8 video streams being processed in parallel, you can jump right to the final demo by directly deploying the deployment manifest in this repo.

## Deploy Deepstream from the Azure Marketplace

We'll start by creating a new IoT Edge solution in VS Code, add the Deepstream module from the marketplace and deploy that to our Jetson Nano.

Note that you could also find Deepstream's module via the [Azure Marketplace website here](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/nvidia.deepstream-iot). You'll use VS code here since Deepstream is an SDK and typically needs to be tweaked or connected to custom modules to deliver an end-to-end solution at the edge.

In VS Code, from your development machine:

1. Start by creating a new IoT Edge solution:
    1. Open the command palette (Ctrl+Shift+P)
    2. Select `Azure IoT Edge: New IoT Edge Solution`
    3. Select a parent folder
    4. Give it a name.
    5. Select `Empty Solution` (if prompted, accept to install iotedgehubdev)
	
2. Add the Deepstream module to your solution:
    1. Open the command palette (Ctrl+Shift+P)
    2. Select `Azure IoT Edge: Add IoT Edge module`
    3. Select the default deployment manifest (deployment.template.json)
    4. Select `Module from Azure Marketplace`.
    5. It opens a new tab with all IoT Edge module offers from the Azure Marketplace. Select the `Nvidia Deepstream SDK` one, select the NVIDIA DeapStream SDK 4.0.2 for Jetson plan and select the `latest` tag.

![Deepsteam in Azure Marketplace](./assets/DeepstreamInMarketplace.png "NVIDIA Deepstream in Azure Marketplace")

3. Deploy the solution to your device:
    1. `Generate IoT Edge Deployment Manifest` by right clicking on the deployment.nano.template.json file
    2. `Create Deployment for Single Device` by right clicking on the generated file in the /config folder
    3. Select your IoT Edge device

4. Start monitoring the messages sent from the device to the cloud
    1. Right-click on your device (bottom left corner)
    2. Select `Start Monitoring Built-In Event Endpoint`

After a little while, (enough time for IoT Edge to download and start DeepStream module which is 1.75GB and compile the AI model), you should be able to see messages sent by the Deepstream module to the cloud via the IoT Edge runtime in VS Code. These messages are the results of Deepstream processing a sample video and analyzing it with an sample AI model that detects people and cars in this video and sends a message for each object found.

![Telemetry sent to IoT Hub](./assets/Telemetry.png "Messages sent from Deepstream module to Azure IoT Hub via the IoT Edge runtime")

## View the processed videos

We'll now modify the configuration of the Deepstream application and the IoT Edge deployment manifest to be able to see the output video streams. We'll do that by asking Deepstream to output the inferred videos to an RTSP video stream and visualize this RTSP stream with VLC.

1. Create your updated Deepstream config file on your Nano device:
    a. Open an SSH connection on your Nano device (for instance from VS Code terminal):
    
    ```cmd
    ssh dlinano@your-nano-ip-address
    ```

    2. Create a new folder to host modified Deepstream config files

    ```bash
    cd /var
    sudo mkdir deepstream
    mkdir ./deepstream/custom_configs
    sudo chmod -R 777 /var/deepstream
    cd ./deepstream/custom_configs
    ```

    3. Use your favorite text editor to create a copy of the sample Deepstream configuration file:
         - Create and open a new file:

        ```bash
        nano test5_config_file_src_infer_azure_iotedge_edited.txt
        ```

        - Copy the content of the original Deepstream configuration file which you can find in this repo under `test5_config_file_src_infer_azure_iotedge.txt`

    4. Edit the configuration file:
        - Disable the first sink (FakeSink) and add a new RTSP sink with the following properties:
        
        ```bash
        [sink0]
        enable=0
        ```

        ```bash
        [sink3]
        enable=1
        #Type - 1=FakeSink 2=EglSink 3=File 4=RTSPStreaming
        type=4
        #1=h264 2=h265
        codec=1
        sync=0
        bitrate=4000000
        # set below properties in case of RTSPStreaming
        rtsp-port=8554
        udp-port=5400
        ```

        - Reduce the number of inferences to be every 3 frames (see `interval` property) otherwise the Nano will drop some frames. In the next section, we'll use a Nano specific config to process 8 video streams in real-time:

        ```
        [primary-gie]
        enable=1
        gpu-id=0
        batch-size=4
        ## 0=FP32, 1=INT8, 2=FP16 mode
        bbox-border-color0=1;0;0;1
        bbox-border-color1=0;1;1;1
        bbox-border-color2=0;1;1;1
        bbox-border-color3=0;1;0;1
        nvbuf-memory-type=0
        interval=2
        ```

        - To make it easier to connect to the output RTSP stream, let's set DeepStream to continuously loop over the test input video files:

        ```
        [tests]
        file-loop=1
        ```

        - Save and Quit (CTRL+O, CTRL+X)

2. Mount your updated config file in the Deepstream module by adding its createOptions in the `deployment.template.json` file from your development's machine:
    - Add the following to your Deepstream createOptions:

    ```json
    "HostConfig":{
        "Binds": ["/var/deepstream/custom_configs:/root/deepstream_sdk_v4.0.2_jetson/sources/apps/sample_apps/deepstream-test5/custom_configs/"]
        }
    ```

    - Edit your Deepstream application working directory and entrypoint to use this updated config file via Deepstream createOptions:
    
    ```json
    "WorkingDir": "/root/deepstream_sdk_v4.0.2_jetson/sources/apps/sample_apps/deepstream-test5/custom_configs/"
    ```

    ```json
    "Entrypoint":["/usr/bin/deepstream-test5-app","-c","test5_config_file_src_infer_azure_iotedge_edited.txt"]
    ```

3. Open the RTSP port of DeepStream module so that you can visualize this feed from another device:
    - Add the following to your Deepstream createOptions, at the root:

    ```json
    "ExposedPorts":{
        "8554/tcp": {}
    }
    ```

    - Add the following to your Deepstream createOptions, in the `HostConfig` node:

    ```json
    "PortBindings": {
        "8554/tcp": [
            {
            "HostPort": "8554"
            }
        ]
    }
    ```

4. Deploy your updated IoT Edge solution:
    1. `Generate IoT Edge Deployment Manifest` by right clicking on the deployment.template.json file
    2. `Create Deployment for Single Device` by right clicking on the generated file in the /config folder
    3. Select your IoT Edge device
    4. Start monitoring the messages sent from the device to the cloud by right clicking on the device (bottom left corner) and select `Start Monitoring Built-In Event Endpoint`

5. Finally, open the default output RTSP stream generated by DeepStream with VLC:
    1. Open VLC
    2. Go to `Media` > `Open Network Stream`
    3. Paste the default `RTSP Video URL` generated by deepstream,  which follows the format `rtsp://your-nano-ip-address:8554/ds-test`
    4. Click `Play`

You should now see messages recevied by IoT Hub via in VS Code AND see the processed video cia VLC.

![Default Output of Deepstream](./assets/4VideosProcessedVLC.png "Output of default Deepstream application running on IoT Edge")

## Process and view 8 video streams (1080p 30fps)

We'll now update Deepstream's configuration to process 8 video streams concurrently (1080p 30fps).

We'll start by updating the batch-size to 8 instead of 4 (`primagy-gie` / `batch-size` property). Then because Tthe Jetson Nano isn't capable of doing inferences on 240 frames per second with a ResNet10 model, we will instead run inferences every 5 frames (`primagy-gie` / `interval` property) and use Deepstream's built-in tracking algorithm for in-between frames, which is less computationnally intensive (`tracker` group). We'll also use a slightly lower inference resolution (defined via `primagy-gie` / `config-file` property). These changes are captured in the Deepstream configuration file below specific to Nano.

1. Update your previously edited Deepstream config file:
    - Open your previous config file:

    ```bash
    nano test5_config_file_src_infer_azure_iotedge_edited.txt
    ```

    - Copy the content of Deepstream's configuration file named `test5_config_file_src_infer_azure_iotedge_nano_8sources.txt` from this repo

    - Save and Quit (CTRL+O, CTRL+X)

2. To simulate 8 video cameras, download and add to Deepstream 8 videos files
    - Open an ssh connection on your Nano device (password=`dlinano`):

    ```cmd
    ssh dlinano@device-ip-address
    ```

    - Host these video files on your local disk

    ```bash
    cd /var/deepstream
    mkdir custom_streams
    sudo chmod -R 777 /var/deepstream
    cd ./custom_streams
    ```

    - Download the video files

    ```bash
    wget -O cars-streams.tar.gz --no-check-certificate "https://onedrive.live.com/download?cid=0C0A4A69A0CDCB4C&resid=0C0A4A69A0CDCB4C%21588371&authkey=AAavgrxG95v9gu0"
    ```

    - Un-compress the video files

    ```bash
    tar -xzvf cars-streams.tar.gz
    ```

    - Mount these video streams by adding the following binding via the `HostConfig` node of Deepstream's  createOptions:

    ```json
    "Binds": [
            "/var/deepstream/custom_configs/:/root/deepstream_sdk_v4.0.2_jetson/sources/apps/sample_apps/deepstream-test5/custom_configs/",
            "/var/deepstream/custom_streams/:/root/deepstream_sdk_v4.0.2_jetson/sources/apps/sample_apps/deepstream-test5/custom_streams/"
            ]
    ```

3. Verify that your are still using your updated configuration file and still expose Deepstream's RTSP port (8554). You can double check your settings by comparing your deployment file to the one in this repo.
4. To speed up IoT Edge message throughput, configure the edgeHub to use an in-memory store. In your deployment manifest, set the `usePersistentStorage` environment variable to `false` in edgeHub configuration (next to its `settings` node) and disable unused protocol heads (DeepStream uses MQTT to communicate with the EdgeHub):

    ```json
    "edgeHub": {
                    "env": {
                      "usePersistentStorage": {
                        "value": "false"
                      },
                      "amqpSettings__enabled": {
                        "value": false
                      },
                      "httpSettings__enabled": {
                        "value": false
                      }
                    }
    ```

5. Deploy your updated IoT Edge solution:
    1. `Generate IoT Edge Deployment Manifest` by right clicking on the deployment.template.json file
    2. `Create Deployment for Single Device` by right clicking on the generated file in the /config folder
    3. Select your IoT Edge device

6. Finally, wait a few moments for DeepStream to restart and open the default output RTSP stream generated by DeepStream with VLC:
    1. Open VLC
    2. Go to `Media` > `Open Network Stream`
    3. Paste the default `RTSP Video URL` generated by deepstream,  which follows the format `rtsp://your-nano-ip-address:8554/ds-test`
    4. Click `Play`

You should now see the 8 video streams being processed and displayed via VLC.

![8 video streams processed real-time](./assets/8VideosProcessedVLC.png "8 video streams processed in real-time by a Jetson Nano with Deepstream and IoT Edge")

## Use your own AI model with Custom Vision

Finally, let's use a custom AI model instead of DeepStream's default one. We'll take the use case of a soda can manufaturer who wants to improve the efficienty of its plant by detecting soda cans that fell down on production lines.
We'll use simulated cameras to monitor each of the lines, collect images, train a custom AI model with [Custom Vision](https://www.customvision.ai/) which is a no-code computer vision AI model builder, to detects cans that are up or down and then deploy this custom AI model to DeepStream.

1. Let's start by creating a new Custom Vision project in your Azure subscription:

    - Go to [http://customvision.ai](https://www.customvision.ai/)
    - Sign-in
    - Create a new Project
    - Give it a name like `Soda Cans Down`
    - Pick up your resource, if none select `create new` and select `SKU - F0` (F0 is free) or (S0)
    - Select `Project Type` = `Object Detection`
    - Select `Domains` = `General (Compact)`

We've already collected training images for you. [Download this compressed folder](https://1drv.ms/u/s!AEzLzaBpSgoMo_R2), unzip it and upload the training images to Custom Vision.

2. We then need to label all of them:

    - Click on an image
    - Label the cans that are up as `Up` and the ones that are down as `Down`
    - Hit the right arrow to move on to the next image and label the remaining 70+ images...or read below to use a pre-built model with this set of images

![Labelling in Custom Vision](./assets/CV-Labelling.png "Labelling in Custom Vision")

3. Once you're done labeling, let's train and export your model:
    - Train your model by clicking on `Train`
    - Export it by going to the `Performance` tab, clicking on `Export` and choosing `ONNX`
    - `Download` your custom AI model and unzip it

4. Finally, we'll deploy this custom vision model to the Jetson Nano and configure DeepStream to use this model.

    - Open an ssh connection on your Nano device (password=`dlinano`):
    
        ```cmd
        ssh dlinano@device-ip-address
        ```
    
    - Create a folder to store your custom model:
    
        ```bash
        cd /var/deepstream
        mkdir custom_models
        sudo chmod -R 777 /var/deepstream
        cd ./custom_models
        ```
    
    - Copy this custom model to your Jetson Nano, either by copying your own model with `scp` or by using this pre-built one:
    
        ```bash
        wget -O cans-onnx-model.tar.gz --no-check-certificate "https://onedrive.live.com/download?cid=0C0A4A69A0CDCB4C&resid=0C0A4A69A0CDCB4C%21588388&authkey=AC4OIGTkjg_t5Cc"
        tar -xzvf cans-onnx-model.tar.gz
        ```
    
    - For DeepStream to understand how to parse the bounding boxes provided by a model from Custom Vision, we need to download an extra library:
    
        ```bash
        wget -O libnvdsinfer_custom_impl_Yolo_Custom_Vision.so --no-check-certificate "https://onedrive.live.com/download?cid=0C0A4A69A0CDCB4C&resid=0C0A4A69A0CDCB4C%21588374&authkey=ADqq__XBNC06kI0"
        ```
    
    - Download raw video streams that we'll use to simulate cameras
    
        ```bash
        cd ../custom_streams
        wget -O cans-streams.tar.gz --no-check-certificate "https://onedrive.live.com/download?cid=0C0A4A69A0CDCB4C&resid=0C0A4A69A0CDCB4C%21588372&authkey=AJfRMnW2qvR3OC4"
        tar -xzvf cans-streams.tar.gz
        ```
    
    - Edit DeepStream configuration file to point to the updated video stream inputs and your custom vision model:
    
        - Open DeepStream configuration file:
        
        ```bash
        cd ../custom_configs
        nano test5_config_file_src_infer_azure_iotedge_edited.txt
        ```
        
        - Copy the content of Deepstream's configuration file named `test5_config_file_src_infer_azure_iotedge_nano_custom_vision.txt` from this repo
        
        - Save and Quit (CTRL+O, CTRL+X)
        - Create another configuration file specific to the inference enfine (which is referenced in the above configuration file):
    
        ```bash
        nano config_infer_custom_vision.txt
        ```
    
        - Copy the content of inference's configuration file named `config_infer_custom_vision.txt` from this repo
        - Double check that the `num-detected-classes` property maps to the number of classes or objects that you've trained your custom vision model for.
        - Save and Quit (CTRL+O, CTRL+X)
        - Create a last configuration file to name your cameras (which is referenced via the `camera-id` property in the main DeepStream configuration file):
    
        ```bash
        nano msgconv_config_soda_cans.txt
        ```
    
        - Copy the content of inference's configuration file named `msgconv_config_soda_cans.txt` from this repo
        - Save and Quit (CTRL+O, CTRL+X)

- Mount these video streams, models, configuration files by adding the following bindings via the `HostConfig` node of Deepstream's  createOptions:

```json
"Binds": [
        "/var/deepstream/custom_configs/:/root/deepstream_sdk_v4.0.2_jetson/sources/apps/sample_apps/deepstream-test5/custom_configs/",
        "/var/deepstream/custom_streams/:/root/deepstream_sdk_v4.0.2_jetson/sources/apps/sample_apps/deepstream-test5/custom_streams/",
        "/var/deepstream/custom_models/:/root/deepstream_sdk_v4.0.2_jetson/sources/apps/sample_apps/deepstream-test5/custom_models/"
        ]
```

6. Deploy your updated IoT Edge solution:
    1. `Generate IoT Edge Deployment Manifest` by right clicking on the deployment.template.json file
    2. `Create Deployment for Single Device` by right clicking on the generated file in the /config folder
    3. Select your IoT Edge device

7. Finally, wait a few moments for DeepStream to restart and open the default output RTSP stream generated by DeepStream with VLC:
    1. Open VLC
    2. Go to `Media` > `Open Network Stream`
    3. Paste the default `RTSP Video URL` generated by deepstream,  which follows the format `rtsp://your-nano-ip-address:8554/ds-test`
    4. Click `Play`

We are now visualizing the processing of 3 real time (e.g. 30fps 1080p) video streams with a custom vision AI models that we built in minutes to detect custom visual anomalies!

![Custom Vision](./assets/sodaCansVLC.png "3 soda cans manufacturing lines are bieing monitored with a custom AI model built with Custom Vision")

## Operating the solution with IoT Central app

Let's create a new IoT Central app to remotely control the Jetson Nano.

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

### Setting up your device to be used with your IoT Central application

We'll start from a blank Jetson installation (Jetpack v4.3), copy a few files locally that are needed for the application such as video files to simulate RTSP cameras and deepstream configuration files, and install IoT Edge.

1. On your Jetson Nano create a folder name `data` at the root:

    ```bash
    sudo mkdir /data
    ```

4. Download and extra setup files in the `data` directory:

    ```bash
    cd /data
    sudo wget -O setup.tar.bz2 --no-check-certificate "https://onedrive.live.com/download?cid=0C0A4A69A0CDCB4C&resid=0C0A4A69A0CDCB4C%21588425&authkey=AAZkjybAWRaCfCc"
    sudo tar -xjvf setup.tar.bz2
    ```

5. Make the folder accessible from a normal user account:

    ```bash
    sudo chmod -R 777 /data
    ```

6. Connect your device to your IoT Central application by editing IoT Edge configuration file:

    - Use your favorite text editor to edit IoT Edge configuration file:

    ```bash
    sudo nano /etc/iotedge/config.yaml
    ```

    - Comment out the "Manual provisioning configuration" section so it looks like this:

    ```bash
    # Manual provisioning configuration
    #provisioning:
    #  source: "manual"
    #  device_connection_string: ""
    ```

    - Uncomment the "DPS symmetric keyi provisioning configuration" and add your IoT Central app's scope id, registration_id which is your device Id and its primary symmetric key:

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

    - Save and exit your editor (Ctrl+O, Ctrl+X)

    - Now Restart the Azure IoT Edge runtime with the following command:

    ```bash
    sudo systemctl restart iotedge
    ```

    - After a few moments the Edge runtime restarts and establishes a connection with your IoT Central application.

### Let's see it in action!

1. Plug your device, press the power button and give it a few minutes to boot. It should automatically connect to the conference's wifi, start its application thanks to IoT Edge and report back its usage and IP address to IoT Central.
2. Connect to your IoT Central application
3. Go to `Devices` and find the device number corresponding to your Jetson Nano (You should have received this number from the proctors and will range from 01-45, for example if you received 01 then your device will be jetson-nano-01)
4. Click on this device and go to the `Dashboard` tab
5. Verify that active telemetry is being sent by the device to IoT Central
6. Copy the `RTSP Video URL` from the `Device` tab
7. Open VLC and go to `Media` > `Open Network Stream` and paste the `RTSP Video URL` copied above as the network URL and click `Play`

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

As a pre-requisite, let's create a new Custom Vision project in your subscription:

- Go to [http://customvision.ai](https://www.customvision.ai/)
- Sign-in
- Create a new Project
- Give it a name like `Soda Cans Down`
- Pick up your resource, if none select `create new` and select `SKU - F0` or (S0)
- Select `Project Type` = `Object Detection`
- Select `Domains` = `General (Compact)`

We then need to collect images to build a custom AI model. In the interest of time, [here](https://1drv.ms/u/s!AEzLzaBpSgoMo-1l) is a set of images that have already been captured for you that you can upload to Custom Vision. Download it, unzip it and upload all the images into your Custom Vision project.

We then need to label all of them:

- Click on one image
- Label the cans that are up as `Up` and the ones that are down as `Down`
- Hit the right arrow to move on to the next image and label the remaining 70+ images...or read below to use a pre-built one with this set of images
- Once you're done labeling, click on `Train`
- To export it, go to the `Performance` tab, click on `Export` and choose `ONNX`
- Right-click on the `Download` button and select `copy link address` to copy the anonymous location of a zip file of your ccustom model

In the interest of time, you can use [this pre-built Custom Vision model](https://onedrive.live.com/download?0C0A4A69A0CDCB4C&resid=0C0A4A69A0CDCB4C%21587636&authkey=AOCf3YsqcZM_3WM).

Finally, we'll deploy this custom vision model to the Jetson Nano using IoT Central. In IoT Central:

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

## Going further

Thank you for attending this workshop! There are other content that you can try with your Jetson Nano at [http://aka.ms/jetson-on-azure](http://aka.ms/jetson-on-azure)!