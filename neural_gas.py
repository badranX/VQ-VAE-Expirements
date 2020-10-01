import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import numpy as np

class VectorQuantizerNG(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay,max_time =100, epsilon=0.001):
        super(VectorQuantizerNG, self).__init__()
        self._yi =  10
        self._yf = 0.01
        self._y = 10
        self._timeMax = max_time
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-10,10)
        #self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        self._decay = decay
        self._epsilon = epsilon
    def forward(self, inputs, time=100):
         # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        flat_input = flat_input.type(torch.FloatTensor)
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight)
        #TODO unflaten by view(input_shape or size check)
        if self.training:
            self._y = self._yi*(self._yf/self._yi)**(time/self._timeMax) 
            with torch.no_grad():
                _ , ordering = torch.sort(distances, descending=False, dim=1)
                ordering = torch.exp(-1*ordering/self._y)
                hv = ordering.t().mm(flat_input)
                sums = ordering.sum(dim=0).unsqueeze(1)
                hw = sums*self._embedding.weight
                self._embedding.weight = nn.Parameter(self._embedding.weight + self._epsilon*(hv - hw))
                #print(self._embedding.weight)
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        # convert quantized from BHWC -> BCHW
        return loss, quantized.contiguous(), perplexity, encodings, self._embedding.weight.detach().numpy(), (self._epsilon*(hv-hw)).detach().numpy()
mean1 = (0.1, 2.)
mean2 = (-30.,0.)
cov = [[1., 0.], [0., 1.]]
group1 = np.random.multivariate_normal(mean1, cov, 80)
group2 = np.random.multivariate_normal(mean2, cov, 90)
data = np.concatenate((group1,group2),axis=0)
data = np.take(data,np.random.permutation(data.shape[0]),axis=0,out=data)
data = torch.from_numpy(data)
NG = VectorQuantizerNG(2,2,0.1,0.1,max_time=20*len(data))
import matplotlib.pyplot as plt
plt.ion()
fig, ax = plt.subplots()
def animate(w,arrow):
    plt.cla()
    ax.scatter(data[:,0],data[:,1], color='C1')
    ax.scatter(w[:,0],w[:,1], color='red')
    ax.quiver(w[:,0],w[:,1],arrow[:,0],arrow[:,1],color='blue')
    plt.pause(0.05)
def train(data, iterations):
    counter = 0
    NG = VectorQuantizerNG(2,2,0.1,0.1,max_time=iterations*len(data))
    for i in range(iterations):
        for j in range(len(data) - 10):
            x = data[j:j+9]
            _ , _,_,_, w, arrow= NG(x,counter)
            counter += 1
            if counter % 10 == 0:
                animate(w,arrow)
train(data,40)


