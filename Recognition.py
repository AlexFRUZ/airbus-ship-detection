# Importing the necessary modules
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QTextEdit, QFileDialog
from PyQt5.QtGui import QPixmap
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# Path to the saved model
model_path = 'detection_ship1.h5'

class MainWindow(QMainWindow):
    # Creating GUI
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Image Classifier")
        self.setGeometry(100, 100, 400, 600)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        self.label = QLabel(self)
        self.label.setFixedSize(400, 400)
        layout.addWidget(self.label)

        self.text_edit = QTextEdit(self)
        self.text_edit.setFixedWidth(400)
        self.text_edit.setReadOnly(True)
        layout.addWidget(self.text_edit)

        load_button = QPushButton('Load Image', self)
        load_button.clicked.connect(self.load_image)
        layout.addWidget(load_button)

        classify_button = QPushButton('Classify', self)
        classify_button.clicked.connect(self.classify_image)
        layout.addWidget(classify_button)

        central_widget.setLayout(layout)

        # Load the model
        self.model = load_model(model_path)

    # A function to load image 
    def load_image(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Open Image', '.', 'Image Files (*.jpg *.png)')
        if filename:
            pixmap = QPixmap(filename)
            pixmap = pixmap.scaledToWidth(400)
            self.label.setPixmap(pixmap)
            self.image_path = filename

    # A function to classify the uploaded image
    def classify_image(self):
        if hasattr(self, 'image_path'):
            img = image.load_img(self.image_path, target_size=(150, 150))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = img / 255.0

            # Prediction
            prediction = self.model.predict(img)
            confidence = prediction[0][0]
            if confidence >= 0.5:
                result = "Ship"
            else:
                result = "No Ship"

            if result == "Ship":
                self.text_edit.setText(f"Prediction: {result}, Confidence: {confidence:.2f}")
            else:
                self.text_edit.setText(f"Prediction: {result}, Confidence: {1 - confidence:.2f}")
        else:
            self.text_edit.setText("Please load an image first.")

# Run the application
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
