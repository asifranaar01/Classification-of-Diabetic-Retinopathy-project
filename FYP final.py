
import Import_data
import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt


def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 5

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(5, 1)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

def import_image(dir):
    # setting valid image types
    valid_image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]  # specify your vald extensions here
    valid_image_extensions = [item.lower() for item in valid_image_extensions]

    # importing train image
    imageDIR = dir
    image_path_list = []
    image_arr = []

    for file in os.listdir(imageDIR):
        extension = os.path.splitext(file)[1]
        if extension.lower() not in valid_image_extensions:
            continue
        image_path_list.append(os.path.join(imageDIR, file))

    for imagePath in image_path_list:
        image = cv2.imread(imagePath)
        image_arr.append(image)

    return image_arr


#######################################################Assigning weights###############################################################
def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))

######################################################## Creating a convolutional layer ################################################################

def create_convolutional_layer(input, num_input_channels, conv_filter_size, num_filters):

    #We shall define the weights that will be trained using create_weights function.
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    ## We create biases using the create_biases function. These are also trained.
    biases = create_biases(num_filters)

    ## Creating the convolutional layer
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    layer += biases

    ## We shall be using max-pooling.
    layer = tf.nn.max_pool(value=layer,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')
    ## Output of pooling is fed to Relu which is the activation function for us.
    layer = tf.nn.relu(layer)

    return layer


def create_flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer = tf.reshape(layer, [-1, num_features])

    return layer


def create_fc_layer(input,
                    num_inputs,
                    num_outputs,
                    use_relu=True):
    # Let's define trainable weights and biases.
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    acc = sess.run(accuracy, feed_dict=feed_dict_train)
    val_acc = sess.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))

def train(num_iteration):
    global total_iterations

    for i in range(total_iterations,
                   total_iterations + num_iteration):

        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)

        feed_dict_tr = {x: x_batch,
                        y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch,
                         y_true: y_valid_batch}

        sess.run(optimizer, feed_dict=feed_dict_tr)

        if i % int(data.train.num_examples / batch_size) == 0:
            val_loss = sess.run(cost, feed_dict=feed_dict_val)
            epoch = int(i / int(data.train.num_examples / batch_size))

            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
            save_path = saver.save(sess, 'C:/Users/HP User/Anaconda3/envs/TensorFlowBasics/DR_model')
            print("Model saved in path: %s" % save_path)
    total_iterations += num_iteration

if __name__ == "__main__":

    batch_size = 5
    test_batch_size = 5
    # Prepare input data
    classes = os.listdir('C:/Users/user/Desktop/FYPImplementation/train')
    num_classes = len(classes)

    # 20% of the data will automatically be used for validation
    validation_size = 0.2
    img_size = 128
    num_channels = 3
    train_path = 'C:/Users/user/Desktop/FYPImplementation/train'
    test_path = 'C:/Users/user/Desktop/FYPImplementation/true_test_classes'
    test_classes = os.listdir('C:/Users/user/Desktop/FYPImplementation/true_test_classes')
    img_shape = (img_size, img_size)
    # We shall load all the training and validation images and labels into memory using openCV and use that during training
    data = Import_data.read_train_sets(train_path, img_size, classes, validation_size=validation_size)
    #test_data = Import_data.read_test_sets(test_path, img_size, test_classes, validation_size=validation_size)

    print("Complete reading input data. Will Now print a snippet of it")
    print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
    print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))
    #print("Number of files in Test-set:\t{}".format(len(test_data.test.labels)))

    # creating a placeholder that will hold the input training images.
    x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
    y_true_cls = tf.argmax(y_true, dimension=1)

    # Configuring the CNN

    # Convolutional Layer 1
    filter_size1 = 5
    num_filters1 = 16

    # Convolutional Layer 2
    filter_size2 = 5
    num_filters2 = 36

    # Fully Connected Layer
    fc_size = 128

    # creating the layers
    layer_conv1 = create_convolutional_layer(input=x, num_input_channels=num_channels, conv_filter_size=filter_size1, num_filters=num_filters1)
    layer_conv2 = create_convolutional_layer(input=layer_conv1, num_input_channels=num_filters1, conv_filter_size=filter_size2, num_filters=num_filters2)
    layer_flat = create_flatten_layer(layer_conv2)

    layer_fc1 = create_fc_layer(input=layer_flat, num_inputs=layer_flat.get_shape()[1:4].num_elements(), num_outputs=fc_size, use_relu=True)

    layer_fc2 = create_fc_layer(input=layer_fc1, num_inputs=fc_size, num_outputs=num_classes, use_relu=False)

    # getting the probability distribution of each class by applying softmax to the output of the fclayer2
    y_pred = tf.nn.softmax(layer_fc2, name="y_pred")
    # the class having the higher probability is the choice of prediction
    y_pred_cls = tf.argmax(y_pred, dimension=1, name="y_pred_cls")

    # define the cost that will be minimized to reach the optimum value of weights.
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                            labels=y_true)
    cost = tf.reduce_mean(cross_entropy)

    # gradient calculation and weight optimization
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.Session()
    total_iterations = 0
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    train(num_iteration=20)







