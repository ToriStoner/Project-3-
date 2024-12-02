# ASL Letter Classification using CNN with Edge Detection

## Contents of the Repository

This repository contains a comprehensive analysis aimed at classifying American Sign Language (ASL) letters using a Convolutional Neural Network (CNN) with Sobel edge detection. The goal of this analysis is to create a classification model using CNN with an edge detector feature that can analyze various components of the images to make a prediction of what letter is being signed in the image. To evaluate the effectiveness and accuracy of the model, accuracy, recall, F-1 and precision scores will be calculated. The model will be considered successful if the accuracy score, precision score, recall score, and F-1 score are all above 0.8. 

## Software and Platform

### Software Used
- Python 3.8
- Jupyter Notebook

### Required Packages
- Tensorflow.keras
- NumPy
- Pandas
- Matplotlib
- os
- Shutil
- Random
- PIL
- Scikit-learn

### Platform
This project was developed and tested on:
- Windows 10
- macOS Monterey

## Repository Structure

```
.
├── DATA
│   ├── Data Appendix.pdf
│   ├── test_data.zip
│   ├── train_data.zip
│   └── valid_data.zip
├── OUTPUTS
│   ├── Class Distribution.png
│   ├── Evaluation Metrics for CNN.png
│   ├── Image Height Distribution.png
│   ├── Image Width Distribution.png
│   ├── Random Images from Each Class.zip
│   ├── Training Accuracy over Epochs.png
│   └── Training Loss Over Epochs.png
├── REFERENCES
│   └── DS4002_Project3_References.pdf
├── SCRIPTS
│   ├── ASL_data_EDA.ipynb
│   └── DS4002_CNN_Code.ipynb
├── Tensor Flow
│   ├── DS4002_CNN_Code.ipynb
│   ├── DS4002_CNN_StTART.ipynb
│   ├── ResNet50
│   └── code
├── LICENSE
└── README.md
```

## Instructions for Reproducing Results

1. **Environment Setup**:
   - Install Python 3.8 or later.
   - Install required packages using pip:
     ```
     pip install tensorflow keras numpy pandas matplotlib seaborn opencv-python scikit-learn
     ```

2. **Data Preparation**:
   - Unzip the data files in the `DATA` folder (`test_data.zip`, `train_data.zip`, `valid_data.zip`).
   - Ensure the unzipped data is in the correct directory structure.

3. **Exploratory Data Analysis**:
   - Open and run the `ASL_data_EDA.ipynb` notebook in the `SCRIPTS` folder.
   - This will generate the EDA plots saved in the `OUTPUTS` folder.

4. **Model Training and Evaluation**:
   - Open the `DS4002_CNN_Code.ipynb` notebook in the `SCRIPTS` folder.
   - Run all cells in the notebook sequentially.
   - This will train the CNN model with Sobel edge detection and evaluate its performance.
   - The results and evaluation metrics should be the same as the ones saved in the `OUTPUTS` folder.

5. **Reviewing Results**:
   - Check the `OUTPUTS` folder for generated plots and metrics.
   - The `Evaluation Metrics for CNN.png` will show the model's performance.
   - Training progress can be observed in `Training Accuracy over Epochs.png` and `Training Loss Over Epochs.png`.

6. **Additional Information**:
   - Refer to `Data Appendix.pdf` in the `DATA` folder for detailed information about the dataset.
   - Check `DS4002_Project3_References.pdf` in the `REFERENCES` folder for project references.

Note: Ensure you have sufficient computational resources, as CNN training can be resource-intensive and time consuming. If using a GPU on a remote desktop is possible, it will significantly speed up the training process.
