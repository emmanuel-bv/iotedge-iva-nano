# Visual Anomaly Detection over multiple cameras with NVIDIA Jetson Nano devices workshop

In this workshop, you'll discover how to build a solution that can process several real-time video streams with an AI model on a $100 device, how to build your own AI model to detect custom anomalies and finally how to operate it remotely.

We'll put ourselves in the shoes of a soda can manufacturer who wants to improve the efficienty of its plant. An improvement that he'd like to make is to be able to detect soda cans that fell down on his production lines, monitor his production lines from home and be alerted when this happen. He has 3 production lines, all moving at a fairly quick speed.

To satisfy the real-time, multiple cameras, custom AI model requirements, we'll build this solution using [NVIDIA Deepstream](https://developer.nvidia.com/deepstream-sdk) on a [NVIDIA Jetson Nano device](https://developer.nvidia.com/embedded/buy/jetson-nano-devkit). 
We'll build our own AI model with [Azure Custom Vision](https://www.customvision.ai/). We'll deploy and connect it to the Cloud with [Azure IoT Edge](https://azure.microsoft.com/en-us/services/iot-edge/) and [Azure IoT Central](https://azure.microsoft.com/en-us/services/iot-central/). Azure IoT Central will be used to do the monitoring and alerting.

## Prerequisites

- **Have an Azure account with an Azure Subscription**: Accounts will be provided during the lab. Please use them to go through this lab.

- **Hardware**: You need a [NVIDIA Jetson Nano device](https://developer.nvidia.com/embedded/buy/jetson-nano-devkit) with a [5V-4A barrel jack power supply like this one](https://www.adafruit.com/product/1466), which requires a jumper cable (such as [these ones](https://www.amazon.com/120pcs-Multicolor-Jumper-Arduino-Raspberry/dp/B01BAXKDN4/ref=asc_df_B01BAXKDN4/?tag=hyprod-20&linkCode=df0&hvadid=198075247191&hvpos=1o1&hvnetw=g&hvrand=12715964868364075974&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=9033288&hvtargid=pla-317965496827&psc=1)) on pins J48. See the [Power Guide section of the Jetson Nano Developer Kit](https://developer.nvidia.com/embedded/dlc/jetson-nano-dev-kit-user-guide) for more details. Alternatively, a 5V-2.5A Micro-USB power supply will work without a jumper cable but may limit the performance of your Deepstream application. In all cases, please make sure to use the default `Max` power source mode (e.g. 10W).

![Jetson Nano](./assets/JetsonNano.png "NVIDIA Jetson Nano device used to run Deepstream with IoT Edge")

- **A USB cable MicroB to Type A to connect your Jetson Nano to your developer's machine with the USB Device Mode**: we'll use the USB Device Mode provided in [NVIDIA's course base image](https://courses.nvidia.com/courses/course-v1:DLI+C-IV-02+V1/info). With this mode, you do not need to hook up a monitor directly to your Jetson Nano. Instead, boot your device and wait for 30 seconds then open your favorite browser, go to [http://192.168.55.1:8888](http://192.168.55.1:8888) and enter the password `dlinano` to get access to a command line on your Jetson Nano.

![Jupyter Notebook](./assets/JupyterNotebook.png "Jetson Nano controlled by a Jupyter Notebook via the USB Device Mode")

- **An SSH client to connect your Jetson Nano**: The USB Device Mode terminal is limited because it does not support copy/paste. So to make it easier to go through the steps of this sample, open an SSH connection with your favorite SSH Client, such as Command Prompt.

    1. Find your IP address using the USB Device Mode terminal (your ip address is the "inet" one):

        ```bash
        /sbin/ifconfig eth0 | grep "inet" | head -n 1
        ```

    2. Make sure that your laptop is on the same network as your Jetson Nano device and open an SSH connection on your Jetson Device (password = `dlinano`):

        ```bash
        ssh dlinano@your-ip-address
        ```

- **VLC to view RTSP video streams**: To visualize the output of the Jetson Nano without HDMI screen (there is only one per table), we'll use VLC from your laptop to view a RTSP video stream of the processed videos. [Install VLC](https://www.videolan.org/vlc/index.html) if you dont have it yet.

The next sections walks you step-by-step to deploy Deepstream on an IoT Edge device, update its configuration via a pre-built IoT Central application and build a custom AI model with Custom Vision. It explains concepts along the way.

## Understanding the solution running at the Edge

The soda can manufucturer already asked a partner to build a first protoype solution that can analyze video streams with a given AI model and connect it to the cloud. The solution built by this partner is composed of two main blocks:

1. **NVIDIA DeepStream**, which does all the video processing

 DeepStream is an highly-optimized video processing pipeline, capable of running one ore more deep neural networks, e.g. AI models. It provides outstanding performances thanks to several techniques that we'll discover below. It is a must-have tool whenever you have complex video analytics requirements like real-time object detection or when employing cascading AI models.

DeepStream runs as a container, which can be deployed and managed by IoT Edge. It also is integrated with IoT Edge to send all its outputs to IoT Edge runtime.

The DeepStream application we are using was easy to build since we use the out-of the box once provided by NVIDIA in the [Azure Marketplace here](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/nvidia.deepstream-iot?tab=Overview). We're using this module as-is and are only configuring it from the IoT Central bridge module.

![Deepsteam in Azure Marketplace](./assets/DeepstreamInMarketplace.png "NVIDIA Deepstream in Azure Marketplace")

2. A **bridge to IoT Central**, which transforms telemetry sent by DeepStream into a format understood by IoT Central and configures DeepStream remotely.

It formats all telemetry, properties, and commands using [IoT Plug and Play](https://docs.microsoft.com/en-us/azure/iot-pnp/overview-iot-plug-and-play) aka PnP, which is the declarative language used by IoT Central to understand how to communicate with a device.

### Understanding NVIDIA DeepStream

Deesptream is a SDK based on GStreamer, an open source, battle-tested platform to create video pipelines. It is very modular with its concepts of plugins. Each plugins have `sinks` and `sources`. NVIDIA provides several plugins as part of Deepstream which are optimized to leverage NVIDIA's GPUs or other NVIDIA hardware like dedicated encoding/decoding chips. How these plugins are connected with each others is defined in the application's configuration file.

Here is an example of what an end-to-end DeepStream pipeline looks like:

![NVIDIA Deepstream Application Architecture](./assets/DeepStreamArchitecture.png).

You can learn more about its architecture in [NVIDIA's official documentation](https://docs.nvidia.com/metropolis/deepstream/dev-guide/index.html#page/DeepStream_Development_Guide%2Fdeepstream_app_architecture.html).

To better understand how NVIDIA DeepStream works, let's have a look at its [default configuration file copied here in this repo](DemoModeDeepStreamConfiguration.txt) (called `Demo Mode` in IoT Central UI later on).

Observe in particular:

- The `sources` sections: they define where the source videos are coming from. We're using local videos to begin with and will switch to live RTSP streams later on.
- The `sink` sections: they define where to output the processed videos and the output messages. We use RTSP to stream a video feed out and all out messages are sent to the Azure IoT Edge runtime.
- The `primary-gie` section: it defines which AI model is used to detect objects. It also defines how this AI model is applied. As an example, note the `interval` property set to `4`: this means that inferencing is actually executed only once every 5 frames. Bounding boxes are displayed continuously though because a tracking algorithm, which is computationally less expensive than inferencing, takes over in between. The tracking algorithm used is set in the `tracking` section. This is the kind of out-of-the-box optimizations provided by DeepStream that enables us to process 240 frames per second on a $100 device. Other notable optimizations are using dedicated encoding/decoding hardware, only loading frames in memory once (zero in-memory copy), pushing the vast majority of the processing to GPUs, batching frames from multiple streams, etc.

### Understanding the connection to IoT Central

IoT Edge connects to IoT Central with the regular Module SDK (you can look at the [source code here](https://github.com/ebertrams/iotedge-iva-nano/blob/master/modules/IoTCentralBridge/src/services/iotCentral.ts)). Telemetry, Properties and Commands that the IoT Edge Central bridge module receives/sends follow [IoT Plug and Play](https://docs.microsoft.com/en-us/azure/iot-pnp/overview-iot-plug-and-play) aka PnP format, which is enforced in the Cloud by IoT Central. IoT Central enforces them against a Device Capability Model (DCM), which is a file that defines what this IoT Edge device is capable of doing.

- Click on `Devices` in the left nav of the IoT Central application
- Observe the templates in the second column: they define all the devices that this IoT Central application understands. All the Jetson Nano devices of this workshop are using a version of the `NVIDIA Jetson Nano DCM` device template. In the case of IoT Edge, an IoT Edge deployment manifest is also attached to a DCM version to create a device template. If you want to see the details on how the device template that we use look like, you can look at [this Device Capability Model](https://github.com/ebertrams/iotedge-iva-nano/blob/master/NVIDIAJetsonNanoDcm.json) and at [this IoT Edge deployment manifest](https://github.com/ebertrams/iotedge-iva-nano/blob/master/deployment.template.json).

Enough documentation! Let's now see the solution built by our partner in action.

## Setting up the solution

Let's start by creating a new IoT Central app to remotely control the Jetson Nano.

### Create a new IoT Central app

We'll start from a pre-built template of IoT Central, which already includes a the pre-built Device Capability Model and IoT Edge deployment manigest mentionned above for our video analytics solution running on the NVIDIA Jetson Nano.

- From your browser, go to: https://apps.azureiotcentral.com/build/new/af1fe1b4-d92e-45fc-9d1f-ea4decdc961d
- Login with the following credentials:
    - Username: `lab01_userXX@iotsummit.xyz` where XX = your Jetson Nano number, for instance `01`.
    - Password: Use the one given in the room
- Give a name and URL to your application
- Select `Standard 2` Pricing plan.
- Change the directory to be `Microsoft (microsoft.onmicrosoft.com)`
- Select the `IoT_SubscriptionContainer_4` Azure subscription
- Keep `United States` as your location
- Click on `Create`

### Create an IoT Edge device from your IoT Central app

We'll create a new IoT Edge device in your IoT Central application that will enable the NVIDIA Jetson Nano to connect to IoT Central.

- Go to the `Devices` tab
- Select the `NVIDIA Jetson Nano DCM` device template
- Click on `New`
- Give a name to your device by editing the `Device ID` and the `Device name` fields (let's use the same name for both of these fields in this workshop)
- Click on `Create`
- Click on your new device
- Click on the `Connect` button in the top right corner
- Copy your `ID Scope` value, `Device ID` value and `Primary key` value and save them for later.

### Setting up your device to be used with your IoT Central application

We'll start from a blank Jetson installation (Jetpack v4.3), copy a few files locally that are needed for the application such as video files to simulate RTSP cameras and deepstream configuration files, install IoT Edge and configure it to connect to your IoT Central instance.

1. On your Jetson Nano create a folder name `data` at the root:

    ```bash
    sudo mkdir /data
    ```

2. Download and extra setup files in the `data` directory:

    ```bash
    cd /data
    sudo wget -O setup.tar.bz2 --no-check-certificate "https://onedrive.live.com/download?cid=0C0A4A69A0CDCB4C&resid=0C0A4A69A0CDCB4C%21588528&authkey=ALyYnqySyxh3dBQ"
    sudo tar -xjvf setup.tar.bz2
    ```

3. Make the folder accessible from a normal user account:

    ```bash
    sudo chmod -R 777 /data
    ```

4. Install IoT Edge (instructions copied [from here](https://docs.microsoft.com/en-us/azure/iot-edge/how-to-install-iot-edge-linux) for convenience):

    ```bash
    curl https://packages.microsoft.com/config/ubuntu/18.04/multiarch/prod.list > ./microsoft-prod.list
    sudo cp ./microsoft-prod.list /etc/apt/sources.list.d/
    curl https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > microsoft.gpg
    sudo cp ./microsoft.gpg /etc/apt/trusted.gpg.d/
    sudo apt-get update
    sudo apt-get install iotedge
    ```

5. Connect your device to your IoT Central application by editing IoT Edge configuration file:

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

    - Uncomment the "DPS symmetric key provisioning configuration" (*not the TPM section but the symmetric key one*) and add your IoT Central app's scope id, registration_id which is your device Id and its primary symmetric key:

    > :warning: Beware of spaces since YAML is space sensitive. In YAML exactly 2 spaces = 1 identation and make sure to not have any trailing spaces.

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

    - And let's verify that the connection to the cloud has been correctly established. If it isn't the case, please check your IoT Edge config file.

    ```bash
    sudo systemctl status iotedge
    ```

As you can guess from this last step, behind the scenes IoT Central is actually using [Azure Device Provisioning Service](https://docs.microsoft.com/en-us/azure/iot-dps/about-iot-dps) to provision devices at scale.

With the IoT Edge device connected to the cloud, it can now report back its IP address to IoT Central. Let's verify that it is the case:

1. Go to your IoT Central application
2. Go to `Devices` tab from the left navigation
3. Click on your device
4. Click on its `Device` tab
5. Verify that the `RTSP Video URL` starts with the IP address of your device

After a minute or so, IoT Edge should have had enough time to download all the containers from the Cloud per IoT Central's instructions and DeepStream should have had enough time to start the default video pipeline, called `Demo mode` in IoT Central UI. Let's see how it looks like:

1. In IoT Central, copy the `RTSP Video URL` from the `Device` tab
2. Open VLC and go to `Media` > `Open Network Stream` and paste this URL `rtsp://192.168.55.1:8554/ds-test` (*not the one found above as it seems to misbehave with the room's network switches) copied above as the network URL and click `Play`
3. In IoT Central, go to to the `Dashboard` tab *of your device* (e.g. from the left nav: `Devices` > `your-device` > `Dashboard`)
4. Verify that active telemetry is being sent by the device to IoT Central. In particular, the number of primary detections which are set to `car` by default should map to the objects detected by the 4 cameras.

At this point, you should see 4 real-time video streams being processed to detect cars and people with a Resnet 10 AI model.

![4 video streams processed real-time](./assets/4VideosProcessedRTSP.png "8 video streams processed in real-time by a Jetson Nano with Deepstream and IoT Edge")

## Operating the solution with IoT Central

 To demonstrate how to remotely manage this solution, we'll send a command to the device to change its input cameras. We'll use your live RTSP cameras from the room as a new input camera.

![IoT Central](./assets/IoTCentral.png "IoT Central UI to remotely manage NVIDIA Jetson devices")

### Changing input cameras

Some RTSP cameras have been setup in the room. We'll send instructions to our Jetson Nano via IoT Central to connect to these live cameras and count the number of people in the room.

- Go to the `Manage` tab
- Unselect the `Demo Mode`, which uses several hardcoded video files as input of car traffic
- Update the `Video Stream 1` property:
    - In the `cameraId`, name your camera, for instance `Room Camera 01`
    - In the `videoStreamUrl`, enter the RTSP stream of this camera given in the room
- Keep the default AI model of DeepStream by keeping the value `DeepStream ResNet 10` as the `AI model type`.
- Keep the default `Secondary Detection Class` as `person`
- Hit `Save`

This sends a command to the device to update its DeepStream configuration file with these new properties and to restart DeepStream. If you were still streaming the output of the DeepStream application, this stream will be taken down as DeepStream will restart.

Let's have a closer look at DeepStream configuration to see what has changed compared to the initial `Demo Mode` configuration which is copied [here](DemoModeDeepStreamConfiguration.txt). From your SSH client:

1. Open an SSH connection with your Jetson Nano IP address. Username is `dlinano` and so is the default password.

    ```bash
    ssh dlinano@YOUR_IP_ADDRESS
    ```

2. Open up the default configuration file of DeepStream to understand its structure:

    ```bash
    nano /data/misc/storage/DSConfig.txt
    ```

3. Look after the first `source` and observe how parameteres provided in IoT Central UI got copied here.

Within a minute, DeepStream should restart. You can observe its status in IoT Central via the `Modules` tab. Once `deepstream` module is back to `Running`, copy again the `RTSP Video Url` field from the `Device` tab and give it to VLC (`Media` > `Open Network Stream` > paste the `RTSP Video URL` > `Play`).

You should now detect people from one of the room's camera. The count of `Person`, aka `Secondary Detection` in the `dashboard` tab *of your device* in IoT Central should go up.

We've just remotely updated the configuration of this intelligent video analytics solution!

## Use a custom AI model to detect visual anomalies

We'll use simulated cameras to monitor each of the soda cans production lines and we'll collect images and build a custom AI model to detects cans that are up or down. We'll then deploy this custom AI model to DeepStream via IoT Central. To do a quick Proof Of Concept, we'll use the [Custom Vision service](https://www.customvision.ai/), a no-code computer vision AI model builder.

As a pre-requisite, let's create a new Custom Vision project in your subscription:

- Go to [http://customvision.ai](https://www.customvision.ai/)
- Sign-in
- Create a new Project
- Give it a name like `Soda Cans Down`
- Pick up your resource, if none select `create new` and select `SKU - F0` or (S0)
- Select `Project Type` = `Object Detection`
- Select `Domains` = `General (Compact)`

We then need to collect images to build a custom AI model. In the interest of time, [here](https://1drv.ms/u/s!AEzLzaBpSgoMo-1l) is a set of images that have already been captured for you that you can upload to Custom Vision. Download it, unzip it and upload all the images into your Custom Vision project.

We then need to label our images:

- Click on an image
- Label the cans that are up as `Up` and the ones that are down as `Down`
- Hit the right arrow to move on to the next image and label the remaining 70+ images...or read below to use a pre-built model with this set of images

![Labelling in Custom Vision](./assets/CV-Labelling.png "Labelling in Custom Vision")

Once you're done labeling, let's train and export your model:
- Train your model by clicking on `Train`
- Export it by going to the `Performance` tab, clicking on `Export` and choosing `ONNX`
- Right-click on the `Download` button and select `copy link address` to copy the anonymous location of a zip file of your ccustom model


In the interest of time, you can also use [this link to a pre-built Custom Vision model](https://onedrive.live.com/download?0C0A4A69A0CDCB4C&resid=0C0A4A69A0CDCB4C%21587636&authkey=AOCf3YsqcZM_3WM).

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

## Creating an alert

To be alerted as soon as a soda can is down, we'll set up an alert to send an email whenever a new soda is detected as being down.

With IoT Central, you can easily define rules and alerts based on the telemetry received by IoT Central. Let's create one whenever a soda can is down.

1. Go to the `Rules` tab in the left nav
2. Click on `New`
3. Give it a name like `Soda can down!`
4. Select your device template `NVIDIA Jetson Nano DCM`
5. Create a Condition with the following attributes:
    - Telemetry = `Secondary Detection Count`
    - Operator = `Is greater than`
    - Value = `1` and hit Enter
6. Create an `email` Action with the following attributes:
    - Display name = `Soda can down`
    - To = your email address used to login to your IoT Central application
    - hit `Done`
7. `Save`

In a few seconds, you should be receiving some mails :)

## Clean-up

This is the end of the workshop. Because there will be another session that uses the same device and azure account after you, please clean up the resources you've installed to let others start fresh:

- **Clean up on the Jetson Nano**, via your SSH terminal:

    ```bash
    sudo rm -r /data
    sudo apt-get remove --purge -y iotedge
    ```

- **Deleting your IoT Central application**, from your browser:
    - Go to your IoT Central application
    - Click on the `Administration` tab from the left nav
    - Click on `Delete` the application and confirm

## Going further

Thank you for attending this workshop! We hope that you enjoyed it and found it valuable.

If you want to run this workwhop at home, you can use [this set of instructions](AtHomeWorkshopInstructions.md), which can work with your own resources at home.

There are other content that you can try with your Jetson Nano at [http://aka.ms/jetson-on-azure](http://aka.ms/jetson-on-azure)!