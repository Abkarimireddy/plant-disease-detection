#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt

IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 40

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "PlantVillage",
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

class_names = dataset.class_names

train_size = 0.8
len(dataset) * train_size

train_ds = dataset.take(516)
len(train_ds)

temp_test_ds = dataset.skip(516)
len(temp_test_ds)

val_size = 0.1
len(dataset) * val_size

val_ds = temp_test_ds.take(64)
len(val_ds)

test_ds = temp_test_ds.skip(64)
len(test_ds)

def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shufflesize=10000):
    ds_size = len(ds)
    if shuffle:
        ds = ds.shuffle(shufflesize, seed=12)  # Fixed the variable name here
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
   
    train_ds = ds.take(train_size)
    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size + val_size)  # Adjusted to skip both train and validation data
    return train_ds, val_ds, test_ds

train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.Rescaling(1.0 / 255)
])

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

n_classes = 15
# Define the image size and number of channels
IMAGE_SIZE = 256
CHANNELS = 3

# Define the rescaling factor
rescale_factor = 1.0 / 255.0

# Define the model
model = Sequential([
    Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)),  # Input layer with specified shape
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(n_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
history = model.fit(
    train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    validation_data=val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
)

# Evaluate the model
score = model.evaluate(test_ds)


# In[4]:


history


# In[5]:


history.params


# In[6]:


history.history.keys()


# In[7]:


acc = history.history['accuracy']
val_acc = history.history['accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']


# In[8]:


len(acc)


# In[11]:


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.plot(range(EPOCHS), acc, label='Training Accuracy', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy', color='red')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()


# In[12]:


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))

# Plot training loss
plt.plot(range(EPOCHS), loss, label='Training Loss', color='blue')

# Plot validation loss
plt.plot(range(EPOCHS), val_loss, label='Validation Loss', color='red')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()


# In[13]:


import numpy as np
import matplotlib.pyplot as plt

image, label = next(iter(test_ds))

plt.imshow(image[0].numpy().astype('uint8'))
plt.show()

print("Actual label:", class_names[label[0].numpy()])

prediction = model.predict(image)

predicted_probabilities = prediction[0]

print("Predicted probabilities:", predicted_probabilities)


# In[14]:


import numpy as np

for images_batch, labels_batch in test_ds.take(1):
    first_image = images_batch[0].numpy().astype('uint8')
    first_label = labels_batch[0].numpy()
    
    plt.imshow(first_image)
    plt.show()
    
    print("Actual label:", class_names[first_label])
    
    batch_prediction = model.predict(images_batch)
    
    print("Predicted label:", class_names[np.argmax(batch_prediction[0])])


# In[15]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# Collect all predicted labels for the test set
predicted_labels = []
true_labels = []

for images_batch, labels_batch in test_ds:
    predictions = model.predict(images_batch)
    predicted_labels.extend(np.argmax(predictions, axis=1))
    true_labels.extend(labels_batch.numpy())

# Calculate F1 score
f1 = f1_score(true_labels, predicted_labels, average='weighted')
print("F1 score:", f1)


# In[ ]:





# In[16]:


import numpy as np

# Initialize variables to count TP, FP, TN, FN
TP = 0
FP = 0
TN = 0
FN = 0

# Iterate over the test dataset to make predictions
for images_batch, labels_batch in test_ds:
    predictions_batch = model.predict(images_batch)
    predicted_labels_batch = np.argmax(predictions_batch, axis=1)
    
    # Update TP, FP, TN, FN counts
    for predicted_label, true_label in zip(predicted_labels_batch, labels_batch):
        if predicted_label == true_label:
            if predicted_label == 1:  # Positive class
                TP += 1
            else:  # Negative class
                TN += 1
        else:
            if predicted_label == 1:  # Positive class
                FP += 1
            else:  # Negative class
                FN += 1

# Calculate metrics
accuracy = (TP + TN) / (TP + FN + TN + FP)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * (precision * recall) / (precision + recall)
sensitivity = recall
specificity = TN / (FP + TN)
false_positive_rate = 1 - specificity

# Print the calculated metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall (Sensitivity):", recall)
print("F1 Score:", f1_score)
print("Specificity:", specificity)
print("False Positive Rate:", false_positive_rate)


# In[17]:


plt.figure(figsize=(20,20))
for images, labels in test_ds.take(1):
    for i in range(6):
        ax = plt.subplot(3, 2, i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        predictions = model.predict(np.expand_dims(images[i], axis=0))
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions) * 100
        actual_class = class_names[labels[i]]
        plt.title(f"Actual: {actual_class},\nPredicted: {predicted_class}.\nConfidence: {confidence:.2f}%")
        plt.axis("off")
    plt.show()


# In[29]:


import os

# Directory path
directory = r"C:\Users\Abi Karimireddy\Downloads\deep learning\Training\final"

# Create directory if it doesn't exist
if not os.path.exists(directory):
    os.makedirs(directory)

# Now save your model
model_version = 1
model.save(f"{directory}{model_version}.keras")


# In[30]:


import os

# Directory containing the model files
directory = r"C:\Users\Abi Karimireddy\Downloads\deep learning\Training\final"

# Get a list of all files in the directory
files = os.listdir(directory)

# Extract the numeric part of each filename
file_numbers = [int(i.split('.')[0]) for i in files if i.split('.')[0].isdigit()]

# Find the maximum version number and increment by 1
max_version = max(file_numbers) + 1 if file_numbers else 1

# Save the model with the incremented version number
model.save(f"{directory}/{max_version}.keras")


# In[ ]:




