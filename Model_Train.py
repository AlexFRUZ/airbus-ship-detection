# Importing the necessary modules
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPool2D
from keras.models import Sequential
from sklearn.utils.class_weight import compute_class_weight
from PIL import ImageFile

# Allow to download partially damaged images
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Parameters
img_height, img_width = 150, 150
batch_size = 128

# Path to dataset
data_dir = 'data1'

# Name and path to save the model 
model_save_path = 'detection_ship1.h5'

# Function for to validate the image
def is_valid_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None or img.size == 0:
            return False
        return True
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return False

all_images = []
all_labels = []
invalid_images = []


# A loop that goes through the images in the classes and sorts them into correct and incorrect ones
for label in ['ships', 'no_ships']:
    folder = os.path.join(data_dir, label)
    for filename in os.listdir(folder):
        if filename.endswith('.jpg'):
            image_path = os.path.join(folder, filename)
            if is_valid_image(image_path):
                all_images.append(image_path)
                all_labels.append(label)
            else:
                invalid_images.append(image_path)

# Output of names and number of damaged files
print(f"Find {len(invalid_images)} damaged files:")
for img in invalid_images:
    print(img)

# Convertation labels into numbers
label_encoder = LabelEncoder()
all_labels = label_encoder.fit_transform(all_labels)

# Separation of data into training and test data
X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)

# Converting numeric labels back to tape for use in generators
y_train_str = label_encoder.inverse_transform(y_train)
y_test_str = label_encoder.inverse_transform(y_test)

# Creating image generators to scale pixel values
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Define training image generator
train_generator = train_datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({'filename': X_train, 'class': y_train_str}),
    x_col='filename',
    y_col='class',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

# Define testing image generator
test_generator = test_datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({'filename': X_test, 'class': y_test_str}),
    x_col='filename',
    y_col='class',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

# Compute class weights to handle class imbalance
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

print(f"Class weights: {class_weights}")

# Creating a model and defining its architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPool2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPool2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Model compilation
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model training
model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=test_generator,
    validation_steps=len(test_generator),
    epochs=5,
    class_weight=class_weights
)

# Save the model
model.save(model_save_path)
