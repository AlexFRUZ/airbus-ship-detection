# Importing the necessary modules
import os
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load and prepare images from directory
def load_data(data_dir, image_size):
    X = []
    y = []
    for label in ['ships', 'no_ships']:
        class_dir = os.path.join(data_dir, label)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = load_img(img_path, target_size=image_size)
            img_array = img_to_array(img)
            X.append(img_array)
            y.append(1 if label == 'ships' else 0)
    return np.array(X), np.array(y)

# Directory containing the dataset and image size parameters
data_dir = r'D:\Train\data'
image_size = (150, 150)

# Load data and labels
X, y = load_data(data_dir, image_size)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalize pixel values to be between 0 and 1
X_train = X_train / 255.0
X_test = X_test / 255.0

# Load the pre-trained model
model = load_model('detection_ship1.h5')

# Prepare data generators for training and testing
train_datagen = ImageDataGenerator()
train_generator = train_datagen.flow(X_train, y_train, batch_size=128)

test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow(X_test, y_test, batch_size=128, shuffle=False)

# Predict on test data
y_pred_probs = model.predict(test_generator)
y_pred = np.where(y_pred_probs > 0.5, 1, 0)

# Calculate evaluation metrics
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print evaluation metrics
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Confusion Matrix:\n{conf_matrix}")

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
