import os
import shutil
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

# Define the main directory containing the six folders of images
main_directory = "path/to/main_directory"

# Load the saved .h5 model
model = tf.keras.models.load_model("path/to/model.h5")

# Define the class labels
class_labels = ["class1", "class2", "class3", "class4", "class5", "class6"]

# Create folders for each class label in the correct_classifications directory
correct_dir = os.path.join(main_directory, "correct_classifications")
os.makedirs(correct_dir, exist_ok=True)
for class_label in class_labels:
    folder_path = os.path.join(correct_dir, class_label)
    os.makedirs(folder_path, exist_ok=True)

# Create folders for each class label in the wrong_classifications directory
wrong_dir = os.path.join(main_directory, "wrong_classifications")
os.makedirs(wrong_dir, exist_ok=True)
for class_label in class_labels:
    folder_path = os.path.join(wrong_dir, class_label)
    os.makedirs(folder_path, exist_ok=True)

# Initialize the variables for storing predictions and true labels
predictions = []
true_labels = []

# Iterate over the folders
for i, class_label in enumerate(class_labels):
    folder_path = os.path.join(main_directory, class_label)
    image_files = os.listdir(folder_path)

    # Iterate over the image files in the folder
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)

        # Read and preprocess the image
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

        # Make predictions
        prediction = model.predict(image)
        predicted_class_index = np.argmax(prediction)
        predicted_class_label = class_labels[predicted_class_index]

        # Store the predictions and true labels
        predictions.append(predicted_class_label)
        true_labels.append(class_label)

        # Copy images into separate folders based on confusion matrix
        if predicted_class_label == class_label:
            destination_folder = os.path.join(correct_dir, class_label, predicted_class_label)
        else:
            destination_folder = os.path.join(wrong_dir, class_label, predicted_class_label)

        os.makedirs(destination_folder, exist_ok=True)
        shutil.copy(image_path, destination_folder)

# Display the confusion matrix
confusion_mat = confusion_matrix(true_labels, predictions, labels=class_labels)
print("Confusion Matrix:")
print(confusion_mat)
