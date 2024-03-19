# SETUP
All infos about the LLVIP dataset can be found here: * ```https://bupt-ai-cz.github.io/LLVIP/```
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
  * ```raw_data``` (all jpgs from ```images_thermal_train/data```)
  * ```index.json``` (from ```images_thermal_train```)
* Run ```prepare_dataset_flir.py``` to create the dataset in the correct format