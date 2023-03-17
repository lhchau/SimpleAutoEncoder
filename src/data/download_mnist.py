from tqdm import tqdm
import requests
import gzip
import os
import numpy as np

url_root = "http://yann.lecun.com/exdb/mnist"

file_dict = {
    'train_images':'train-images-idx3-ubyte.gz',
    'train_labels':'train-labels-idx1-ubyte.gz',
    'test_images':'t10k-images-idx3-ubyte.gz',
    'test_labels':'t10k-labels-idx1-ubyte.gz'
}

def download_mnist(url_root, file_dict=None):
    if file_dict is not None:
        mnist_data = list()
        try:
            for i, key in enumerate(file_dict.keys()):    
                fname = file_dict[key]
                url = os.path.join(url_root, fname)
                # @author: lhchau
                fname = os.path.join("src/data", fname)                
                isExist = os.path.exists(fname)
                if not isExist:
                    response = requests.get(url, stream=True)
                    fsize=len(response.content)
                    print(url)
                    with open(fname, 'wb') as fout:
                        for data in tqdm(response.iter_content(), desc=fname, total=fsize):
                            fout.write(data)

                with gzip.open(fname, "rb") as f_in:                
                    if fname.find('idx3') != -1:        
                        mnist_data.append(np.frombuffer(f_in.read(), np.uint8, offset=16).reshape(-1, 28, 28)) #if images        
                    else:                               
                        mnist_data.append(np.frombuffer(f_in.read(), np.uint8, offset=8))  #if labels
            #return mnist_data in a list format ==> [[train_images], [train_labels], [test_images], [test_labels]] 
            return mnist_data
        except Exception as e:
            print("Something went wrong:", e)
    else:
        print("file_dict cannot be None")