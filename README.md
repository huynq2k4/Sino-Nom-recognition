# Sino-Nom Character Recognition - Group 9 
This is the repository of Group 9's mid-term project in Image Processing - INT3404E 20. Given an image of a character extracted from a Sino-Nom document as input, this project aims to classify the image into one of several classes representing several Sino-Nom characters in the provided dataset.
## Contributors
Nguyễn Quang Huy - 22028077\
Nguyễn Quang Huy - 21020204\
Kiều Minh Khuê - 22028067\
Mai Ngọc Duy - 22028255
## Installation
1. Clone this repository
2. Install neccessary libraries
```
pip install -r requirements.txt
```
3. Get root directory of training dataset and validation dataset. The dataset must have the following structure:
```
├── root-directory
│   ├── train
│   └── val
```
4. Run main.py (for example, dataset directory has the root name "root-directory")
```
python main.py --root_dir=root-directory
```

## Structure
```
├── model
│   ├── ResNet.py #ResNet layer of model
│   └── SimCLR.py #contrastive model
├── data_util.py #file to load data
├── main.py #file to run model
```
## Result
Our model accuracy is about 80% for both validation set and test set.
## Reports
For additional details on the methodologies and models employed in this project, please read the following [report](report)
