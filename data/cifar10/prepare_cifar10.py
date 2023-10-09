import os
import numpy as np
import pickle


data_path = os.path.dirname(__file__)

def get_data_CIFAR(subset, data_path=data_path):
    """
    CIFAR data contains the files data_batch_1, data_batch_2, ...,
    as well as test_batch, so you'll need to combine all train batches
    into one batch. Each of these files is a Python "pickled"
    object produced with cPickle. The code below will open up each
    "pickled" object (i.e. each file) and return a dictionary.

    :param subset: string to indicate which subset of data to get ("train" or "test")
    :param data_path: folder containing the CIFAR data
    :return:
        inputs (NumPy array of uint8),
        labels (NumPy array of string),
        label_names (NumPy array of strings)
    """

    ## https://www.cs.toronto.edu/~kriz/cifar.html
    subset = subset.lower().strip()
    assert subset in ("test", "train"), f"unknown data subset {subset} requested"
    data_files = {
        "train": [f"data_batch_{i+1}" for i in range(5)],
        "test": ["test_batch"],
    }[subset]
    data_meta = f"{data_path}/cifar-10-batches-py/batches.meta"
    data_files = [f"{data_path}/cifar-10-batches-py/{file}" for file in data_files]

    # TODO #1:
    #   Pull in all of the data into cifar_dict.
    #   Check the cifar website above for the API to unpickle the files.
    #   Then, you can access the components i.e. 'data' via cifar_dict[b"data"].
    #   If data_files contains multple entries, make sure to unpickle all of them
    #   and concatenate the results together into a single training set.

    data = []
    labels = []
#     cifar_dict[b"data"] = np.empty((0, 3072))
    for file in data_files:
        with open(file, 'rb') as fo:
            batch_dict = pickle.load(fo, encoding='bytes')
        # concatenate each read
        data += list(batch_dict[b"data"])
        labels += list(batch_dict[b"labels"])
    images = np.array(data)
    labels = np.array(labels)
    
    # read in the meta data, which is the binary label names
    label_names_b = None
    with open(data_meta, 'rb') as fo:
        label_names_b = pickle.load(fo, encoding='bytes')[b"label_names"]

    # extract the label info and decode it from binary string to utf-8 string
    label_names = np.array(list(map(lambda x : x.decode("utf-8"), label_names_b)))

    # TODO #2:
    #   Currently, the variable "label" is a list of integers between 0 and 9,
    #     with 0 meaning "airplane", 1 meaning "automobile" and so on.
    #   You should change the label with more descriptive names, given in the
    #   Numpy array variable "label_names" (remember that label_names contains
    #   binary strings and not UTF-8 strings right now)
    #   This variable "label" should be a Numpy array, not a Python list.
    real_label = label_names[label]
    label = real_label
    
    # TODO #3:
    #   You should reshape the input image np.array to (num, width, height, channels).
    #   Currently, it is a 2D array in the shape of (images, flattened pixels)
    #   You should reshape it into (num, 3, 32, 32), because the pickled images are in
    #     three channels(RGB), and 32 pixels by 32 pixels.
    #   However, we want the order of the axis to be in (num, width, height, channels),
    #     with the RGB channel in the last dimension.
    #   We want the final shape to be (num, 32, 32, 3)
    
    # reshape all images from (1, 1024) to (3, 32, 32)
    image = image.reshape(-1, 3, 32, 32)
    # transpose all images from (3, 32, 32) to (32, 32, 3) so the RGB channel is the last dimension
    image = np.transpose(image, (0, 2, 3, 1))

    # DO NOT normalize the images by dividing them with 255.0.
    # With the MNIST digits, we did normalize the images, but not with CIFAR,
    # because we will be using the pre-trained ResNet50 model, which requires
    # the pixel values to be unsigned integer values between 0 and 255.

    return images, labels, label_names


