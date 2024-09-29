import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define new emotion categories
emotion_map = {
    'angry': 'angry',
    'disgust': None,  # Ignore this
    'fear': None,     # Ignore this
    'happy': 'happy',
    'sad': 'sad',
    'surprise': None, # Ignore this
    'neutral': 'calm'
}

# Paths to the dataset
train_dir = 'archive (2)/train'
test_dir = 'archive (2)/test'

# Create directories for the new labels
new_train_dir = 'filtered/train'
new_test_dir = 'filtered/test'

os.makedirs(new_train_dir, exist_ok=True)
os.makedirs(new_test_dir, exist_ok=True)

for emotion in ['happy', 'sad', 'angry', 'calm']:
    os.makedirs(os.path.join(new_train_dir, emotion), exist_ok=True)
    os.makedirs(os.path.join(new_test_dir, emotion), exist_ok=True)

# Function to move images based on the new mapping
def move_images(data_dir, new_data_dir):
    for emotion, mapped_emotion in emotion_map.items():
        if mapped_emotion:
            src_dir = os.path.join(data_dir, emotion)
            dest_dir = os.path.join(new_data_dir, mapped_emotion)
            if os.path.exists(src_dir):
                for img_file in os.listdir(src_dir):
                    src_file_path = os.path.join(src_dir, img_file)
                    dest_file_path = os.path.join(dest_dir, img_file)
                    # Check if the source and destination paths are the same
                    if src_file_path != dest_file_path:
                        shutil.copy(src_file_path, dest_file_path)
                    else:
                        print(f"Skipping file {src_file_path} as it's the same as the destination.")


# Apply the mapping to the train and test sets
move_images(train_dir, new_train_dir)
move_images(test_dir, new_test_dir)
# Set up the data generators
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # Use 20% of training data for validation
)

train_generator = datagen.flow_from_directory(
    new_train_dir,
    target_size=(150, 150),  # Image size used for MobileNet
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    new_train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Testing set
test_generator = datagen.flow_from_directory(
    new_test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)
