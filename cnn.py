import ssl
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
from tensorflow.python.keras import layers, models
import logging
from colorlog import ColoredFormatter
from PIL import Image
import os

# Disable SSL certificate verification (NOT RECOMMENDED ONLY FOR TESTING PURPOSES ONLY).
ssl._create_default_https_context = ssl._create_unverified_context

# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Color formatter
formatter = ColoredFormatter(
    "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
)

# Configure handler with formatter
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def main():
    logger.info("Starting CNN Network")
    logger.info("Loading Dataset CIFAR-10")
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0

    logging.info("Defining model")
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(32, 32, 3)))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(10))
    
    
    logger.info("Compiling model")
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    logger.info("Training model")
    history = model.fit(train_images, train_labels, epochs=10, 
                        validation_data=(test_images, test_labels))
    
    logger.info("Evaluating model")
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    logger.info(f"Model Evaluation has finished. Accuracy: {test_acc}")

    logger.info("Making predictions with test images")
    predictions = model.predict(test_images)

    class_names = ['Avión', 'Automóvil', 'Pájaro', 'Gato', 'Ciervo',
                'Perro', 'Rana', 'Caballo', 'Barco', 'Camión']

    num_images = 5
    indices = np.random.choice(len(test_images), num_images)

    plt.figure(figsize=(20, 10), dpi=500)

    for i, idx in enumerate(indices):
        predicted_label = np.argmax(predictions[idx])
        true_label = test_labels[idx][0]
        confidence = np.max(predictions[idx])
        image = test_images[idx]
        
        img = Image.fromarray((image * 255).astype(np.uint8))

        plt.subplot(1, num_images, i + 1)
        plt.imshow(img, interpolation='spline36')
        plt.axis('off')

        predicted_class_name = class_names[predicted_label]
        true_class_name = class_names[true_label]

        plt.title(f'Predicción: {predicted_class_name} ({confidence:.2f})\nReal: {true_class_name}')
        logger.info(f"Showing prediction for image {idx}: Predicted: {predicted_class_name}, Actual: {true_class_name}")

    save_path = 'predictions.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=500)
    
    abs_save_path = os.path.abspath(save_path)
    logger.info(f"Predictions have been save in: {abs_save_path}")
    
if __name__ == "__main__":
    main()