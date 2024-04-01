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

# DATA PREPROCESSING
* LLVIP images
  * 15k infrared images with persons of which the location in the image is annotated
    * Extraction of the different persons (> 100x100 px) based on the annotations leads to 18k images in total
  * transformation from infrared RGB to grayscale to reduce the number of channels
* Oxford IIIT Pet Dataset
  * 7345 RGB images of dogs and cats
  * transformation from RGB to grayscale to reduce the number of channels
* FLIR ADAS Dataset
  * 12k infrared images of which the location of various items like persons, cars, signs, etc. in the image is annotated
    * Extraction of the different cars (> 100x100 px) based on the 'car' annotations leads to 3.7k images in total 
  * all images are provided in grayscale

* Randomly select the necessary number of images from the cars category to match the number of cats and dogs images.
* Combine and shuffle the datasets to ensure random distribution.
* Split the combined dataset into training, validation, and test sets maintaining the balance between classes.
