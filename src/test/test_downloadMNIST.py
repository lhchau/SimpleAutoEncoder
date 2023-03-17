from ..data.downloadMNIST import *

dataset = download_mnist(url_root,file_dict)
train_images, train_labels, test_images, test_labels = download_mnist(url_root, file_dict)

assert dataset is not None, "Something in dataset went wrong!\n"
assert train_images is not None, "Something in train_images went wrong!\n"
assert train_labels is not None, "Something in train_labels went wrong!\n"
assert test_images is not None, "Something in test_images went wrong!\n"
assert test_labels is not None, "Something in test_labels went wrong!\n"
