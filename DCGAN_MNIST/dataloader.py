"""
dataloader using torchvision for data download
"""

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def MNISTDataset(mode='train'):
    mnist_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(), 
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    if(mode == 'train'):
        return datasets.MNIST(
            root=Path().absolute().parent.parent / 'mnist_data',
            train=False,
            transform=mnist_transform,
            download=True
        )
    elif(mode == 'test'):
        return datasets.MNIST(
            root=Path().absolute().parent.parent / 'mnist_data',
            train=True,
            transform=mnist_transform,
            download=True
        )
    else:
        raise Exception("unexpected type")

class MNISTDataloader(object):
    """ dataloader for MNIST """
    def __init__(self, dataset, opt):
        super().__init__()
        kwargs = {'num_workers': opt.num_workers} if opt.gpu else {}

        self.data_loader = DataLoader(
            dataset, batch_size=opt.batch_size, shuffle=True, **kwargs)

        self.data_iter = self.data_loader.__iter__()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num-workers', type=int, default=4)
    parser.add_argument('-e', '--epoch', type=int, default=40)
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    opts = parser.parse_args()

    train_dataset = MNISTDataset('train')
    train_data_loader = MNISTDataloader(train_dataset, opts)

    test_dataset = MNISTDataset('test')
    test_data_loader = MNISTDataloader(test_dataset, opts)

    print('[+] Size of the train dataset: %05d, train dataloader: %04d' \
        % (len(train_dataset), len(train_data_loader.data_loader)))   
    print('[+] Size of the test dataset: %05d, test dataloader: %04d' \
        % (len(test_dataset), len(test_data_loader.data_loader)))

    for _n, (images, label) in enumerate(train_data_loader.data_loader):
        print(images, label)
