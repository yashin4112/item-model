# train_model.py

import tensorflow as tf
import os

# Step 1: Set paths for training and validation data directories
train_dir = 'data/train'
val_dir = 'data/val'

# Step 2: Use ImageDataGenerator to load and augment images
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet.preprocess_input,  # Preprocessing input with MobileNet's function
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet.preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

print(train_generator.class_indices)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Step 3: Load MobileNet model pre-trained on ImageNet
base_model = tf.keras.applications.MobileNet(
    weights='imagenet', include_top=False, input_shape=(224, 224, 3)
)

# Step 4: Add custom classification layers
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
predictions = tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')(x)

# Combine the base MobileNet model and custom layers into one model
model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

# Step 5: Freeze the base layers and train only the custom classification head
for layer in base_model.layers:
    layer.trainable = False

# Compile the model with Adam optimizer and categorical cross-entropy loss
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss='categorical_crossentropy', metrics=['accuracy'])

# Step 6: Train the model
history = model.fit(
    train_generator,
    epochs=10,  # Adjust the number of epochs as needed
    validation_data=val_generator
)

# Step 7: Unfreeze last few layers of the base model and fine-tune
for layer in base_model.layers[-30:]:  # Unfreeze the last 30 layers
    layer.trainable = True

# Recompile with a lower learning rate for fine-tuning
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
              loss='categorical_crossentropy', metrics=['accuracy'])

# Step 8: Fine-tune the model
history_finetune = model.fit(
    train_generator,
    epochs=10,  # Adjust the number of epochs for fine-tuning
    validation_data=val_generator
)

# Step 9: Save the fine-tuned model
model.save('old_fine_tuned_mobilenet_grocery.h5')

print("Model training and fine-tuning completed. Model saved as 'fine_tuned_mobilenet_grocery.h5'.")
