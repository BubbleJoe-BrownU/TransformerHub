#!/bin/bash

echo "Downloading CIFAR-10 dataset..."

curl -k "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz" --output cifar-10-python.tar.gz
tar xzvf cifar-10-python.tar.gz
rm cifar-10-python.tar.gz