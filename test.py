import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
image_batch, label_batch = test_dataset.as_numpy_iterator().next()
predictions = model(image_batch)

# Apply a sigmoid since our model returns logits
predictions = tf.nn.sigmoid(predictions)
predictions = tf.where(predictions < 0.5, 0, 1)

print('Predictions:\n', predictions.numpy())
print('Labels:\n', label_batch)

plt.figure(figsize=(10, 10))
for i in range(9):
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(image_batch[i].astype("uint8"))
  plt.title(class_names[predictions[i][0]])
  plt.axis("off")