# fiat-lux

## Problem
Nowadays, motion detectors recognise movement quite reliably. Unfortunately, they do not differentiate between motion sources. The aim of this project is to control a light source only when both a movement is detected and the source is a person.

This means that the light no longer switches on when the neighbour's cat sneaks through the garden at night.

## Solution
With the help of a Raspberry Pi, a motion sensor and an IR camera, the test setup is to be designed so that the motion sensor triggers the camera as soon as it detects movement. The IR image will then be analysed using a pre-trained Tensorflow light model. If the model comes to the conclusion that it is a person, the light is switched on.

## Training Data Sources
Based on the pre-trained model imagenet (https://paperswithcode.com/dataset/imagenet), the model was re-trained with the images from https://paperswithcode.com/dataset/llvip. The model was then verified using real IR images from different situations.

## Hardware setup
The hardware used was the Raspberry Pi 5 with the Raspberry Pi Camera 3 NoIR and the RPI HC-SR501 IR sensor. The images were also saved on the SSD card to enable visual assessment and additional external verification outside of the Raspberry Pi.
