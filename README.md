# Catintell
The deployment of  'Universal Cataract Fundus Image Restoration Model Learning from Unpaired High-quality and Cataract Images'. This model means to improve the quality of cataract images and make the lesions and vessels clear to ophthalmologists. Catintell has a unique dual-branch degradation-restoration structure and can use fully unpaired HQ-cataract data to train a restoration model. The rest of the codes will be released upon acceptance.

ODIR/Kaggle/Catintell datasets are tested to show its generality.

## Usage

For now, the Catintell project provides inference codes and all of the model files. To use this model, please install the dependency with the following command.
```
pip install -r requirement.txt
```
Then, download the pre-trained weights for the model: [googledrive](https://drive.google.com/file/d/14fVDHBoSjkv30ZB5GiAWXqcunIdLWd5v/view?usp=drive_link) and save it under the ```./pretrained``` file

The sample images are already provided in ```./datasets/testimages```, you can also add your own cataract image data to test images.

Use this command to run the test:
```
CUDA_VISIBLE_DEVICES= xxxx  python ca_fin/test.py -opt ./ca_fin/options/folder_test_cat.yml
```
Then, the results can be found in ```./results``` dictionary.
