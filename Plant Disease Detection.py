# ── Step 1: Clone the official PlantVillage GitHub repo ──────────────────────
!git clone https://github.com/spMohanty/PlantVillage-Dataset.git

# ── Step 2: Check what's inside ───────────────────────────────────────────────
import os

base_path = '/content/PlantVillage-Dataset/raw/color'
classes   = sorted(os.listdir(base_path))

print(f"Total classes : {len(classes)}")
print(f"\nClass names:")
for i, c in enumerate(classes):
    count = len(os.listdir(os.path.join(base_path, c)))
    print(f"  {i+1:2}. {c:45s} → {count} images")
  # ── Download 300 MB zip directly ──────────────────────────────────────────────
!wget -q --show-progress \
  "https://data.mendeley.com/public-files/datasets/tywbtsjrjv/files/d5652a28-c1d8-4b76-97f3-72fb80f94efc/file_downloaded" \
  -O /content/plantvillage.zip

# ── Extract ────────────────────────────────────────────────────────────────────
!unzip -q /content/plantvillage.zip -d /content/plant_data
!ls /content/plant_data
# ── Install ────────────────────────────────────────────────────────────────────
!pip install -q tensorflow-datasets

import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import os

# ── Download automatically (~300 MB, takes 2–3 minutes) ───────────────────────
(train_ds, val_ds, test_ds), info = tfds.load(
    'plant_village',
    split=['train[:70%]', 'train[70%:85%]', 'train[85%:]'],
    with_info=True,
    as_supervised=True
)

CLASS_NAMES  = info.features['label'].names
NUM_CLASSES  = info.features['label'].num_classes
TOTAL_IMAGES = info.splits['train'].num_examples

print(f"Total images  : {TOTAL_IMAGES:,}")
print(f"Total classes : {NUM_CLASSES}")
print(f"Train size    : {int(TOTAL_IMAGES * 0.70):,}")
print(f"Val size      : {int(TOTAL_IMAGES * 0.15):,}")
print(f"Test size     : {int(TOTAL_IMAGES * 0.15):,}")
print(f"\nClass names (first 10):")
for i, name in enumerate(CLASS_NAMES[:10]):
    print(f"  {i+1}. {name}")
  # ── Preprocess & batch (plug directly into your training code) ────────────────
IMG_SIZE   = (160, 160)
BATCH_SIZE = 64
AUTOTUNE   = tf.data.AUTOTUNE

def preprocess(image, label):
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_saturation(image, 0.8, 1.2)
    return image, label

train_dataset = (
    train_ds
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .map(augment,    num_parallel_calls=AUTOTUNE)
    .shuffle(1000)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

val_dataset = (
    val_ds
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

test_dataset = (
    test_ds
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

print("Datasets ready!")
print(f"Train batches : {len(train_dataset)}")
print(f"Val batches   : {len(val_dataset)}")
print(f"Test batches  : {len(test_dataset)}")
# ── Visualize sample images ────────────────────────────────────────────────────
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 8))
sample_batch = next(iter(train_dataset.take(1)))
images, labels = sample_batch

for i in range(12):
    plt.subplot(3, 4, i+1)
    plt.imshow(images[i].numpy())
    plt.title(CLASS_NAMES[labels[i].numpy()].replace('___', '\n'),
              fontsize=7)
    plt.axis('off')

plt.suptitle('PlantVillage — sample training images', fontsize=13)
plt.tight_layout()
plt.show()
 # ── Build model (same MobileNetV2 as before) ───────────────────────────────────
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, callbacks
import tensorflow as tf

base = MobileNetV2(weights='imagenet', include_top=False,
                   input_shape=(*IMG_SIZE, 3))
base.trainable = False

inputs  = tf.keras.Input(shape=(*IMG_SIZE, 3))
x       = base(inputs, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.BatchNormalization()(x)
x       = layers.Dense(256, activation='relu')(x)
x       = layers.Dropout(0.4)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

# ── Phase 1: train head ────────────────────────────────────────────────────────
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',   # use sparse because tfds gives integer labels
    metrics=['accuracy']
)

cb = [
    callbacks.EarlyStopping(monitor='val_accuracy', patience=4,
                            restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                                patience=2, verbose=1),
    callbacks.ModelCheckpoint('/content/best_model.keras',
                              save_best_only=True, monitor='val_accuracy')
]

print("Phase 1: training head only...")
hist1 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    callbacks=cb
)

# ── Phase 2: fine-tune top 30 layers ──────────────────────────────────────────
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nPhase 2: fine-tuning...")
hist2 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=15,
    callbacks=cb
)
# ── Evaluate on test set ───────────────────────────────────────────────────────
test_loss, test_acc = model.evaluate(test_dataset, verbose=1)
print(f"\nTest Accuracy : {test_acc*100:.2f}%")
print(f"Test Loss     : {test_loss:.4f}")
