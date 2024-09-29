import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths to the dataset
train_dir = 'archive (2)/train_filtered'
test_dir = 'archive (2)/test_filtered'

# Set up the data generators
batch_size = 8
datagen = ImageDataGenerator(
    rescale=1./255,
    
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='categorical'
)

# Load the pre-trained MobileNet model without the top layers
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Add custom layers for emotion classification (4 classes: happy, sad, angry, calm)
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)  # Adding dropout
output_layer = Dense(4, activation='softmax')(x)

# Create the new model
model = Model(inputs=base_model.input, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
try:
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        epochs=30,  # Reduce the number of epochs for testing
    )
except Exception as e:
    print(f"An error occurred: {e}")

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")

# Save the fine-tuned model
model.save('fine_tuned_mood_recognition_model.keras')
