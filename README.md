# alexnet
**Python implementation of "Alexnet: ImageNet Classification with Deep Convolutional Networks" (Krizhevsky et al.)**

Read the paper [here!](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)

A majority of the data processing code (how to load images, etc.) was taken from [this article](https://www.digitalocean.com/community/tutorials/alexnet-pytorch).

## Paper implementation overview
My implementation of AlexNet is done in a combination of NumPy and PyTorch. I've implemented only pieces relevant to AlexNet __exclusively__ in NumPy (as their own `nn.Module`), and the rest has been implemented via PyTorch (Linear layers, etc.)

The NumPy implementation includes three key components:

- **CNN**: A convolutional layer (similar to torch.nn.Conv2d)
- **max_pooling**: A max pooling layer (similar to torch.nn.MaxPool2d)
- **ReLU**: A ReLU activation function (like torch.nn.ReLU)

You can view the implementation of each one in `layers.py` ([here](https://github.com/RollingRo11/alexnet/blob/main/layers.py))

Additionally, I've implemented `torch.nn.Dropout` in the multilayer perceptron (as also done by the authors of the paper) in order to reduce overfitting and boost weight learnings- (I've already implemented Dropout on its own, you can check it out [here!](https://github.com/RollingRo11/dropout))
