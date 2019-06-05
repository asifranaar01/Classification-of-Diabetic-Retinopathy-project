import numpy as np
import keras
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras.metrics import categorical_crossentropy
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.layers.core import Dense, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
#import Data_Import
from keras.applications import vgg16
from scipy import ndimage, misc
import os
import cv2
import itertools
#import Preprocessing
from keras.preprocessing.image import ImageDataGenerator,img_to_array, load_img


def CNN_model():
    # building a sequential model
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=2, strides=1, padding='same', activation='relu', input_shape=(img_size, img_size, num_channels)))
    model.add(Flatten())
    model.add(Dense(3, activation='softmax'))

    # using Adam Optimizer
    optimizer = Adam(lr=1e-3)
    train_epocs = len(train_batches) // train_batch_size
    valid_epocs = len(valid_batches) // valid_batch_size
    # compiling the model
    model.compile(optimizer=optimizer, loss = 'categorical_crossentropy', metrics=['accuracy'])
    # Need to check whether it is an improvement
    # model.compile(loss= 'binary_crossenrtopy', optimizer= 'rmsprop', metrics=['accuracy'])
    # training the model -> verbosity = how we want the output of the training to be displayed
    model.fit_generator(train_batches, validation_data=valid_batches, epochs=5, verbose=3)
    model.summary()
    return model

def VGG16_model():
    print("Training Model... \n")
    #vgg16_model = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
    vgg16_model = vgg16.VGG16()
    #vgg16.summary()
    # modifying the model to classify only 3 classes rather than 1000 classes
    model = Sequential()
    # iterate through the layers in the VGG16 model and add them into the new Sequential model
    for layer in vgg16_model.layers:
        model.add(layer)
    # taking out the last dense layer
    model.layers.pop()
    # excluding the models layer from future training so that the base weights are not updated
    for layer in model.layers:
        layer.trainable = False
    # adding the last dense layer so that the model now can classify 3 classes
    model.add(Dense(3, activation='softmax'))
    steps = len(imgs) // train_batch_size
    # using Adam Optimizer
    optimizer = Adam(lr=1e-3)
    model.compile(optimizer = optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit_generator(train_batches, validation_data=valid_batches, epochs=5, verbose=2)
    print("Model finished training. \n")
    return model


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')


    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return plt.show()


def test_model():
    print("Testing the model...")
    # prediction
    predictions = model.predict_generator(test_batches, steps=test_steps, verbose=0)

    print("Testing complete.")
    return predictions

def set_confusion_matrix(predictions):
    print("Confusion Matrix: ")
    cm_plot_labels = ['Mild DR', 'No DR', 'Severe DR']
    cm = confusion_matrix(test_labels.argmax(axis=1), predictions.argmax(axis=1))
    return cm, cm_plot_labels

if __name__ == "__main__":

    # Prepare input data
    classes = os.listdir('C:/Users/user/Desktop/FYPImplementation/train')
    num_classes = len(classes)
    # 20% of the data will automatically be used for validation
    validation_size = 0.2
    img_size = 224
    num_channels = 3
    train_path = 'C:/Users/user/Desktop/FYPImplementation/train'
    valid_path = 'C:/Users/user/Desktop/FYPImplementation/valid'
    test_path = 'C:/Users/user/Desktop/FYPImplementation/true_test_classes'
    img_shape = (img_size, img_size)
    train_batch_size = 10
    valid_batch_size = 4
    test_batch_size = 5

    # We shall load all the training and validation images and labels into memory using openCV and use that during training
    #data = Data_Import.read_train_sets(train_path, img_size, classes, validation_size=validation_size)

    train_batches = ImageDataGenerator().flow_from_directory(train_path,
                                         target_size=(img_size, img_size), classes=classes, batch_size=train_batch_size)
    valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(img_size, img_size), classes=classes, batch_size=valid_batch_size)
    test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(img_size, img_size), classes=classes, batch_size=test_batch_size)

    imgs, labels = next(train_batches)


    test_images, test_labels = next(test_batches)
    test_classes = test_batches.classes
    test_steps = len(test_batches) // test_batch_size  # to get all the test images from the data


    # # calling the VGG16 model for training
    model = VGG16_model()
    #
    # # test model
    model_test = test_model()
    #
    score = model.evaluate(test_images, test_labels, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    #
    cm, cm_plot_labels = set_confusion_matrix(model_test)
    #
    #
    # model = CNN_model()
    #
    #
    plot_confusion_matrix(cm, classes=cm_plot_labels, title="Confusion Matrix")




