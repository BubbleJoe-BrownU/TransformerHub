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

    data = []
    labels = []
    for file in data_files:
        with open(file, 'rb') as fo:
            batch_dict = pickle.load(fo, encoding='bytes')
        # concatenate each read
        data += list(batch_dict[b"data"])
        labels += list(batch_dict[b"labels"])
    images = np.array(data)
    labels = np.array(labels)
    print(labels[:10])
    
    # read in the meta data, which is the binary label names
    label_names_b = None
    with open(data_meta, 'rb') as fo:
        label_names_b = pickle.load(fo, encoding='bytes')[b"label_names"]

    # extract the label info and decode it from binary string to utf-8 string
    label_names = np.array(list(map(lambda x : x.decode("utf-8"), label_names_b)))

    
    # reshape all images from (1, 1024) to (3, 32, 32)
    images = images.reshape(-1, 3, 32, 32)
    # transform to channel-last if needed
    # image = np.transpose(image, (0, 2, 3, 1))
    images = images.astype(np.float32) /255.

    return images, labels, label_names


if __name__ == "__main__":
    # for debug use
    images, labels, label_names = get_data_CIFAR(subset="train")
    print(len(images))
    print(label_names)
    images, labels, label_names = get_data_CIFAR(subset="test")
    print(len(images))
    print(label_names)