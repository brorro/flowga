from typing import Callable
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset


def get_mnist_train(path, train=True) -> Dataset:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            [0.1307],
            [0.3081]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomResizedCrop(
            size=28, scale=(0.8, 1)),
        transforms.RandomErasing(scale=(0.01, 0.15))])
    mnist = torchvision.datasets.MNIST(
        root=path, train=train, download=True, transform=transform)

    return mnist


def get_mnist_test(path, train=False) -> Dataset:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            [0.1307],
            [0.3081])])
    mnist = torchvision.datasets.MNIST(
        root=path, train=train, download=True, transform=transform)

    return mnist


def get_fashion_mnist_train(path, train=True) -> Dataset:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            [0.2860],
            [0.3530]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()])
    fashion_mnist = torchvision.datasets.FashionMNIST(
        root=path, train=train, download=True, transform=transform)

    return fashion_mnist


def get_fashion_mnist_test(path, train=False) -> Dataset:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            [0.2860],
            [0.3530])])
    fashion_mnist = torchvision.datasets.FashionMNIST(
        root=path, train=train, download=True, transform=transform)

    return fashion_mnist


def get_cifar10_train(path, train=True) -> Dataset:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            [0.49139968, 0.48215841, 0.44653091],
            [0.24703223, 0.24348513, 0.26158784]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomResizedCrop(
            size=32, scale=(0.8, 1)),
        transforms.RandomErasing(scale=(0.01, 0.15))])
    cifar10 = torchvision.datasets.CIFAR10(
        root=path, train=train, download=True, transform=transform)

    return cifar10


def get_cifar10_test(path, train=False) -> Dataset:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            [0.49139968, 0.48215841, 0.44653091],
            [0.24703223, 0.24348513, 0.26158784])])
    cifar10 = torchvision.datasets.CIFAR10(
        root=path, train=train, download=True, transform=transform)

    return cifar10


def get_dataset_fn(dataset_name: str) -> tuple[Callable[[str], Dataset], Callable[[str], Dataset]]:
    if dataset_name == 'cifar10':
        fn = get_cifar10_train, get_cifar10_test
    elif dataset_name == 'mnist':
        fn = get_mnist_train, get_mnist_test
    elif dataset_name == 'fashion_mnist':
        fn = get_fashion_mnist_train, get_fashion_mnist_test
    else:
        raise ValueError(f'Invalid dataset name: {dataset_name}')

    return fn