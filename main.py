import torch
import torch.nn as nn
from data import get_train_valid_loader, get_test_loader
from model import AlexNet


train_loader, valid_loader = get_train_valid_loader(
    data_dir="./data", batch_size=64, augment=False, random_seed=1
)

test_loader = get_test_loader(data_dir="./data", batch_size=64)

num_classes = 10
num_epochs = 20
batch_size = 64
learning_rate = 0.005

model = AlexNet(num_classes)

optimizer = torch.optim.SGD(
    model.parameters(), lr=learning_rate, weight_decay=0.005, momentum=0.9
)


criterion = nn.CrossEntropyLoss()
# Train the model
total_step = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(
        "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
            epoch + 1, num_epochs, i + 1, total_step, loss.item()
        )
    )

    # Validation
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valid_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs

        print(
            "Accuracy of the network on the {} validation images: {} %".format(
                5000, 100 * correct / total
            )
        )
