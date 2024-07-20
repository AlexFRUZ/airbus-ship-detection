# Ship Detection Model

## Description

This project uses Convolutional Neural Networks (CNNs) to detect ships in images. The model is trained on images labeled as either "ships" or "no_ships". The project includes image validation and processing, as well as training a model for image classification.

## Project Structure

- **`data1/`**: Directory containing image data with subdirectories for `'ships'` and `'no_ships'`.
- **`detection_ship1.h5`**: Saved model after training.
- **`Model_Train.py`**: Script for training the model.

## Functionality

1. **Image Validation**:
   - The script checks if images are valid (not corrupted) using OpenCV.
   - Corrupted files are listed in the console for potential removal or recovery.

2. **Data Loading and Processing**:
   - Images are loaded from the directory and split into training and test sets.
   - Image labels are encoded into numerical values.

3. **Data Generators**:
   - `ImageDataGenerator` is used to normalize images and create data generators for training and testing.

4. **Class Weight Calculation**:
   - Class weights are computed to balance the influence of different classes during training.

5. **Model Training**:
   - A CNN model is trained on the training data and evaluated on the test data.
   - The model uses the `'adam'` optimizer and `'binary_crossentropy'` loss function.

6. **Model Saving**:
   - After training, the model is saved to the file `detection_ship1.h5`.

## Usage Instructions

1. **Setup**:
   - Ensure all necessary libraries are installed, including `tensorflow`, `keras`, `opencv-python`, `numpy`, `pandas`, `scikit-learn`.

2. **Run the Script**:
   - Execute the script `Model_Train.py` with the following command:
     ```
     python Model_Train.py
     ```

3. **Check Results**:
   - After training completes, check the results in the file `detection_ship1.h5`.

## Notes

- Make sure the `data1` directory contains subdirectories `'ships'` and `'no_ships'` with appropriate images.
- The model is trained on images resized to 150x150 pixels. You may adjust these parameters if needed.
