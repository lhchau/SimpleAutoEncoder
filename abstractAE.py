import numpy as np
import math
import matplotlib.pyplot as plt

class AbstractAutoEncoder: 
    def __init__(self, encoder, decoder) -> None:
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(x)
        return out
    
    def show_reconstructions(self, x=None, n_cols=5):
        out = self.forward(x)
        out = out.detach().cpu().numpy()
        out = np.squeeze(out)
        
        x_np = x.cpu().numpy()
        x_np = np.squeeze(x_np)
        
        n_images = len(x)
        n_rows = math.ceil(n_images/n_cols)
        fig = plt.figure(figsize=(2*n_cols*1.5, n_rows*1.5))
        plt.axis("off")
        for i in range(n_images):
            plt.subplot(n_rows, 2*n_cols, 2*i+1)
            plt.imshow(x_np[i], cmap="gray")
            plt.xlabel("real")
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.subplot(n_rows, 2*n_cols, 2*i+2)
            plt.imshow(out[i], cmap="gray")
            plt.xlabel("reconstruct")
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)