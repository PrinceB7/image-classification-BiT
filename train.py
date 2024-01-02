import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import tensorflow_datasets as tfds

# Constants
SEEDS = 42
RESIZE_TO = 384
CROP_TO = 224
BATCH_SIZE = 64
STEPS_PER_EPOCH = 10
AUTO = tf.data.AUTOTUNE 
NUM_CLASSES = 5
SCHEDULE_LENGTH = 500
SCHEDULE_BOUNDARIES = [200, 300, 400]

SCHEDULE_LENGTH = SCHEDULE_LENGTH * 512 / BATCH_SIZE

# Set seeds
np.random.seed(SEEDS)
tf.random.set_seed(SEEDS)

# Disable progress bar
tfds.disable_progress_bar()

def load_datasets():
    return tfds.load(
        "tf_flowers",
        split=["train[:85%]", "train[85%:]"],
        as_supervised=True,
    )

def preprocess_train(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.resize(image, (RESIZE_TO, RESIZE_TO))
    image = tf.image.random_crop(image, (CROP_TO, CROP_TO, 3))
    image = image / 255.0
    return (image, label)

def preprocess_test(image, label):
    image = tf.image.resize(image, (RESIZE_TO, RESIZE_TO))
    image = image / 255.0
    return (image, label)

def create_pipelines(train_ds, validation_ds, schedule_length):
    dataset_num_train_examples = train_ds.cardinality().numpy()
    repeat_count = int(
        schedule_length * BATCH_SIZE / dataset_num_train_examples * STEPS_PER_EPOCH
    )
    repeat_count += 50 + 1

    pipeline_train = (
        train_ds.shuffle(10000)
        .repeat(repeat_count)
        .map(preprocess_train, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )

    pipeline_validation = (
        validation_ds.map(preprocess_test, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )
    
    return pipeline_train, pipeline_validation

def create_model(num_classes):
    bit_model_url = "https://tfhub.dev/google/bit/m-r50x1/1"
    bit_module = hub.KerasLayer(bit_model_url)

    class MyBiTModel(keras.Model):
        def __init__(self, num_classes, module, **kwargs):
            super().__init__(**kwargs)
            self.num_classes = num_classes
            self.head = keras.layers.Dense(num_classes, kernel_initializer="zeros")
            self.bit_model = module

        def call(self, images):
            bit_embedding = self.bit_model(images)
            return self.head(bit_embedding)

    return MyBiTModel(num_classes=num_classes, module=bit_module)

def compile_and_train(model, pipeline_train, pipeline_validation, schedule_length):
    learning_rate = 0.003 * BATCH_SIZE / 512
    lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=SCHEDULE_BOUNDARIES,
        values=[
            learning_rate,
            learning_rate * 0.1,
            learning_rate * 0.01,
            learning_rate * 0.001,
        ],
    )
    optimizer = keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

    train_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=6, restore_best_weights=True
        )
    ]

    history = model.fit(
        pipeline_train,
        batch_size=BATCH_SIZE,
        epochs=int(schedule_length / STEPS_PER_EPOCH),
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=pipeline_validation,
        callbacks=train_callbacks,
    )
    return history

def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.plot(hist.history["loss"])
    plt.plot(hist.history["val_loss"])
    plt.title("Training Progress")
    plt.ylabel("Accuracy/Loss")
    plt.xlabel("Epochs")
    plt.legend(["train_acc", "val_acc", "train_loss", "val_loss"], loc="upper left")
    plt.show()


if __name__ == '__main__':
    
    # train pipeline
    train_ds, validation_ds = load_datasets()
    pipeline_train, pipeline_validation = create_pipelines(train_ds, validation_ds, SCHEDULE_LENGTH)
    model = create_model(NUM_CLASSES)
    history = compile_and_train(model, pipeline_train, pipeline_validation, SCHEDULE_LENGTH)
    plot_hist(history)
    
    try: model.save('models/gpt')
    except Exception as e: print(e)
    
    try: model.save('gpt')
    except Exception as e: print(e)
    
    try: model.save_weights('models/model_weights', save_format='tf')
    except Exception as e: print(e)
    
    # accuracy report
    accuracy = model.evaluate(pipeline_validation)[1] * 100
    print("Accuracy: {:.2f}%".format(accuracy))
    

