# fiat lux

## Problem
Nowadays, motion detectors recognise movement quite reliably. Unfortunately, they do not differentiate between motion sources. The aim of this project is to control a light source only when both a movement is detected and the source is a person.

This means that the light no longer switches on when the neighbour's cat sneaks through the garden at night.

## Solution
With the help of a Raspberry Pi, a motion sensor and an IR camera, the test setup is to be designed so that the motion sensor triggers the camera as soon as it detects movement. The IR image will then be analysed using a pre-trained Tensorflow light model. If the model comes to the conclusion that it is a person, the light is switched on.

## Training Data Sources
Based on the pre-trained model imagenet (https://paperswithcode.com/dataset/imagenet), the model was re-trained with the images from https://paperswithcode.com/dataset/llvip, https://universe.roboflow.com/joseph-nelson/thermal-dogs-and-people, https://sites.google.com/view/elizabethbondi/dataset and https://www.flir.com/oem/adas/adas-dataset-form/ . The model was then verified using real IR images from different situations.

## Hardware setup
The hardware used was the Raspberry Pi 5 with the Raspberry Pi Camera 3 NoIR and the RPI HC-SR501 IR sensor. The images were also saved on the SSD card to enable visual assessment and additional external verification outside of the Raspberry Pi.

## Development
### Installing tensor flow on Raspberry Pi
https://pimylifeup.com/raspberry-pi-tensorflow-lite/

# SETUP
* All infos about the LLVIP dataset can be found here: * ```https://bupt-ai-cz.github.io/LLVIP/```
## Preparation
* Create a folder ```dataset```
## Download LLVIP dataset
* Create a folder ```llvip```
* Create a subfolder ```raw_data``` in this folder
* Download the dataset from ```https://github.com/bupt-ai-cz/LLVIP/blob/main/download_dataset.md```
* Unzip to ```raw_data``` which should contain all the images and annotations in the following structure:
    * ```raw_data```
        * ```Annotations```
        * ```infrared```
            * ```test```
            * ```train```
        * ```visible```
            * ```test```
            * ```train```
* Run ```prepare_dataset_llvipi.py``` to create the dataset in the correct format

## Download the FLIR ADAS dataset
* Create a folder ```flir```
* Create a subfolder ```raw_data``` in this folder
* Download the dataset from ```https://www.flir.com/oem/adas/adas-dataset-form/#anchor29```
* Unzip to ```raw_data``` which should contain all the images and annotations in the following structure:
    * ```raw_data/data``` (all jpgs from ```images_thermal_train/data```)
    * ```index.json``` (from ```images_thermal_train```)
* Run ```prepare_dataset_flir.py``` to create the dataset in the correct format