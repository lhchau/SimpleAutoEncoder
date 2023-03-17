from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np

data_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307, ), std=(0.3081, ))
    ]
)

class MNISTCustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __getitem__(self, idx):
        label = self.labels[idx]
        image = self.images[idx]      
        image = self.transform(np.array(image))
        return image, label
    
    def __len__(self):
        return len(self.labels)

# assert dataset is not None, "Something went wrong!\n"

