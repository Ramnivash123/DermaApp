import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

train_dir = "data/train"
test_dir = "data/test"

# Augmentation with more transformations
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8, 1.2]
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Class Labels
classes = [
    "Acne and Rosacea Photos",
    "Eczema Photos",
    "Hair Loss Photos Alopecia and other Hair Diseases",
    "Herpes HPV and other STDs Photos",
    "Melanoma Skin Cancer Nevi and Moles",
    "Nail Fungus and other Nail Disease",
    "Psoriasis Pictures Lichen Planus and related Diseases",
    "Scabies Lyme Disease and other Infestations and Bites"
]

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=32, class_mode="categorical", classes=classes
)

validation_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(224, 224), batch_size=32, class_mode="categorical", classes=classes
)

# Load Pre-trained VGG16 Model
input_shape = (224, 224, 3)
base_model = VGG16(input_shape=input_shape, include_top=False, weights="imagenet")

# Freeze base layers initially
for layer in base_model.layers:
    layer.trainable = False

# Adding new classification head
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(8, activation='softmax')(x)

model = Model(base_model.input, x)

# Compile Model
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks: Reduce LR and Early Stopping
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train Phase 1: Only New Layers
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=[reduce_lr, early_stop]
)

# Fine-Tune Phase: Unfreeze deeper layers and re-train
for layer in base_model.layers[-10:]:  # Unfreeze last 10 layers
    layer.trainable = True

# Recompile with lower LR for fine-tuning
model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

history_fine = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=[reduce_lr, early_stop]
)

# Save Model
model.save("derm_model.h5")
