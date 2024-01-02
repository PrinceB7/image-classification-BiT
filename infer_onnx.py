import cv2
import numpy as np

import onnxruntime
import tensorflow as tf
import tensorflow_datasets as tfds

# Load and preprocess your test data
# Here, using TensorFlow Datasets as an example
# Replace with your own data loading and preprocessing
test_data = tfds.load('tf_flowers', split="train[85%:]")
test_images, test_labels = [], []
for example in test_data:
    # Preprocess the data (reshape, normalize, etc.)
    image, label = example['image'], example['label']
    # Assuming the model expects 224x224 RGB images
    image = tf.image.resize(image, [224, 224])
    image = image / 255.0
    test_images.append(image.numpy())
    test_labels.append(label.numpy())

# Convert to appropriate numpy array shape for ONNX model
test_images = np.array(test_images, dtype=np.float32)


# Load the ONNX model
sess = onnxruntime.InferenceSession('models/flowers.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

# Perform inference
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
preds = sess.run([label_name], {input_name: test_images[:500]})[0]


softmax_preds = tf.nn.softmax(preds, axis=1)
predicted_classes = np.argmax(softmax_preds, axis=1)
accuracy = np.mean(predicted_classes == test_labels[:500])
print(f'Accuracy 2: {accuracy:.3f}')
