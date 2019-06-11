# Written By Mark Dhruba Sikder (26529548) & Asif Rana (27158632)

import keras
from keras import layers
from keras.models import Sequential, Model
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.applications.inception_v3 import InceptionV3
from keras.layers import *
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import os
import itertools
from keras.preprocessing.image import ImageDataGenerator,img_to_array, load_img
from hyperopt import STATUS_OK


def load_preprocessed_images_for_training(img_size):
    """
    This function allows us to load the preprocessed images and then use the images to find the labels and then set up the
    image so that it can be passed to the model.
    :param img_size:
    :return:
    """
    print("Loading Preprocessed images...")
    # Prepare input data
    classes = os.listdir('C:/Users/fit3162-03/AppData/Local/Continuum/anaconda3/envs/tensorflow/FYP_final_Mark&Rana/train_final_process')

    # train directory
    train_path = 'C:/Users/fit3162-03/AppData/Local/Continuum/anaconda3/envs/tensorflow/FYP_final_Mark&Rana/train_final_process'
    # batches
    train_batch_size = 20
    valid_batch_size = 12

    # calling the generator which will split the data
    datagen = ImageDataGenerator(validation_split=0.2)

    # using  the train path to generate the training images
    train_generator = datagen.flow_from_directory(
        train_path,
        target_size=(img_size, img_size),
        batch_size=train_batch_size,
        class_mode='categorical',
        color_mode="rgb", subset="training")

    train_images, train_labels = next(train_generator)
    print("Length of training images: ", len(train_images))

    # using the 20% data in the train directory as the validation set.
    validation_generator = datagen.flow_from_directory(
        train_path,
        target_size=(img_size, img_size),
        batch_size=valid_batch_size,
        class_mode='categorical',
        color_mode="rgb",
        shuffle=False, subset="validation")

    print("Loading complete\n")

    return train_generator, validation_generator, train_batch_size, train_labels, train_images


def VGG16_model(img_size, train_generator, validation_generator):

    print("Training Model... \n")
    base_model = keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))

    # Check the trainable status of the individual layers

    # Create the model
    model = Sequential()
    for layer in base_model.layers:
        model.add(layer)

    # Add new layers
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation='relu'))
    # dropout layer
    model.add(layers.Dropout(0.7))
    model.add(layers.Dense(3, activation='softmax'))

    # freeze the first 18 layers of the model
    for layer in model.layers[:18]:
        layer.trainable = False


    print(model.summary())

    sgd = optimizers.SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    callbacks = [ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1)]

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=35,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples / validation_generator.batch_size,
        verbose=2, callbacks=callbacks)

    plot_training_validation_curve(history)

    return model


def Inception_V3(img_size, train_generator, validation_generator):
    """
    This is the Inception model used for training.
    :param img_size: image size of 299
    :param train_generator: training images with labels
    :param validation_generator: validation image with labels
    :param train_batch_size:
    :param train_images:
    :return:
    """

    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_size,img_size,3))

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer and the dropout layers
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(3, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # reeze the first 249 convolutional layers
    for layer in model.layers[:249]:
        layer.trainable = False

    # call back set to change the validation accuracy as soon as the validation accuracy starts to remain constant
    callbacks = [
        ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=3, verbose=1, mode='auto', cooldown=0,
                          min_lr=1e-6)
    ]

    # sgd optimizer
    sgd = optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # fit the model
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples / train_generator.batch_size,
        epochs=60,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples / validation_generator.batch_size,
        verbose=2, callbacks= callbacks)

    plot_training_validation_curve(history)

    return model


def plot_training_validation_curve(history):
    """
    This function helps to plot the training and validation curve as well as the loss
    :param history: the models records
    :return:
    """
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))
    # Accuracy curves
    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    # Loss curves
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Learning curve loss')
    plt.legend()

    plt.show()


def train_VGG16_model(img_size, train_generator, validation_generator):
    """
    This function calls the function for training VGG16
    :param img_size: size of the image
    :param train_generator: the train images and labels
    :param validation_generator: validation images and labels
    :return: trained model
    """
    model = VGG16_model(img_size, train_generator, validation_generator)
    return model

def train_Inception_V3_model(img_size, train_generator, validation_generator):
    """
    This function calls the function for training InceptionV3
    :param img_size: size of the image
    :param train_generator: the train images and labels
    :param validation_generator: validation images and labels
    :return: trained model
    """
    model = Inception_V3(img_size, train_generator, validation_generator)
    return model


def load_test_data(img_size):
    """
    This function loads the test data
    :return:
    """
    print("Loading test data...")
    classes = os.listdir('C:/Users/fit3162-03/AppData/Local/Continuum/anaconda3/envs/tensorflow/FYP_final_Mark&Rana/test_reza_final')
    test_path = 'C:/Users/fit3162-03/AppData/Local/Continuum/anaconda3/envs/tensorflow/FYP_final_Mark&Rana/test_reza_final'
    test_batch_size = 52
    test_generator = ImageDataGenerator().flow_from_directory(test_path, target_size=(img_size, img_size), classes=classes, batch_size=test_batch_size, color_mode="rgb", shuffle=False)

    test_images, test_labels = next(test_generator)
    print("Load complete.")
    print("Length of test images: ", len(test_images))
    return test_images, test_labels, test_generator

def test_model(model, test_images, test_generator):
    """
    This function helps to make the final predictions of test images
    :param model:
    :param test_images:
    :param test_generator:
    :return:
    """

    print("Testing the model...")
    # prediction
    predictions = model.predict_generator(test_generator, steps=len(test_generator))
    predictions = np.argmax(predictions, axis=-1)  # multiple categories
    predicted_labels = [test_generator.classes[pred_y] for pred_y in predictions]
    print("Testing complete.")
    print("Number of predictions: ", len(predictions))

    return predictions, predicted_labels

def evaluate_model(model, test_labels, test_images):
    """
    This function evaluates the models performance
    :param model: trained model
    :param test_generator: test images and labels
    :return: accuracy
    """
    score = model.evaluate(test_images,test_labels, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1]*100)
    return {'loss': -score[1], 'status': STATUS_OK, 'model': model}

def calculate_fusion(vgg16_model_test, Inception_model_test):
    """
    This function takes two models predictions and
    :param vgg16_model_test:
    :param Inception_model_test:
    :return:
    """
    fused_list = []

    for i in range(len(vgg16_model_test)):

        o1 = vgg16_model_test[i]
        o2 = Inception_model_test[i]

        new_list = [0] * len(o1)

        for j in range(len(o1)):
            new_list[j] = (o1[j] + o2[j]) / 2

        fused_list.append(new_list)

    return fused_list

def eval(fused_list,test_labels):
    """
        This function allows us to make the final evaluation combining both the models
        :param fused_list:
        :param test_labels:
        :return:
        """

    correct = 0
    wrong  = 0
    for i in range(len(fused_list)):

        if(fused_list[i].index(max(fused_list[i])) == test_labels[i].index(max(test_labels[i]))):
            correct += 1
        else:
            wrong +=1
    print(correct,'------', wrong)
    return (correct / (correct + wrong)) * 100


def set_confusion_matrix(predictions, test_labels):
    """
    Sets the confusion matrix
    :param predictions:
    :param test_labels:
    :return:
    """
    print("Confusion Matrix: ")
    cm_plot_labels = ['Mild DR', 'Moderate DR', 'Severe DR']
    cm = confusion_matrix(test_labels.argmax(axis=1), predictions.argmax(axis=1))
    return cm, cm_plot_labels

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function helps to plot the confusion matrix.
    :param cm:
    :param classes: number of classes
    :param normalize: normalize the output
    :param title: titile of the graph
    :param cmap: clolur of each clok
    :return: confusion matrix
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

    plt.show()


def run():
    """
    Run the main program
    :return: results
    """
    # -----------------------------------------------------------------------------TRAINING MODEL---------------------------------------------------------------------------------------------------------------#

    # loading images for vgg16
    train_generator, validation_generator, train_batch_size, train_labels, train_images = load_preprocessed_images_for_training(224)
    # loading images for Inceptionv3
    train_generator_inception, validation_generator_inception, train_batch_size_inception, train_labels_inception, train_images_inception = load_preprocessed_images_for_training(299)
    # train vgg16
    vgg16_model = train_VGG16_model(224, train_generator, validation_generator)
    # train inceptionv3
    inceptionv3_model = train_Inception_V3_model(299, train_generator_inception, validation_generator_inception)

    # -----------------------------------------------------------------------------------TESTING MODEL----------------------------------------------------------------------------------------------------------#

    # load test data
    test_images_vgg16, test_labels_vgg16, test_generator_vgg16 = load_test_data(224)
    test_images_inception, test_labels_inception, test_generator_inception = load_test_data(299)

    # Evaluation
    print("For VGG16 model ", end="    ")
    evaluate_model(vgg16_model, test_labels_vgg16, test_images_vgg16)
    print("")

    model_test_predictions_vgg16, predicted_labels_vgg16 = test_model(vgg16_model, test_images_vgg16, test_generator_vgg16)
    # print("Predictions Inception: ", model_test_predictions_Inception)

    print("For Inception ", end="    ")
    evaluate_model(inceptionv3_model, test_labels_vgg16, test_images_inception)
    print("")

    model_test_predictions_Inception, predicted_labels_inception = test_model(inceptionv3_model, test_images_inception, test_generator_inception)

    print(model_test_predictions_vgg16)
    print(model_test_predictions_Inception)

    ## -------------------------------------------------------------------------FOR ANALYZING PURPOSES-------------------------------------------------------------------------------------------------------------------------##

    #print(predicted_labels_inception)
    # fused_list = calculate_fusion(model_test_predictions_Alexnet.tolist(), model_test_predictions_Inception.tolist())
    # print(eval(fused_list, test_labels.tolist()))
    # confusion matrix vgg16
    #cm, cm_plot_labels = set_confusion_matrix(model_test_predictions_vgg16, test_labels_vgg16)
    #plot_confusion_matrix(cm, classes=cm_plot_labels, title="Confusion Matrix")

    # confusion matrix inception
    #cm, cm_plot_labels = set_confusion_matrix(model_test_predictions_Inception, test_labels_inception)
    #plot_confusion_matrix(cm, classes=cm_plot_labels, title="Confusion Matrix")


if __name__ == "__main__":
    run()
