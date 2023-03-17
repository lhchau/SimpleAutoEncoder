from ..downloadMNIST import *

dataset = download_mnist(url_root,file_dict)

assert dataset is not None, "Something went wrong!\n"
