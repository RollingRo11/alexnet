# alexnet
**Python implementation of Alexnet: ImageNet Classification with Deep Convolutional Networks (Krizhevsky et al.)**

Read the paper [here!](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)

A majority of the baseline code (including how to load images, etc.) was taken from [this](https://www.digitalocean.com/community/tutorials/alexnet-pytorch) article.

## Paper implementation overview
My personal implementation of AlexNet is done in a combination of NumPy and PyTorch. I've implemented only pieces relevant to AlexNet myself in NumPy (as their own `nn.Module`), and the rest has been implemented via PyTorch.

The implementation includes three key components:

- **CNN**: A convolutional layer (similar to torch.nn.Conv2d)
- **max_pooling**: A max pooling layer (similar to torch.nn.MaxPool2d)
- **ReLU**: A ReLU activation function (like torch.nn.ReLU)
