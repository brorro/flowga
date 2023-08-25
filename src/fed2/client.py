from collections import OrderedDict
from functools import partial
from math import fsum
from typing import Callable, Optional, Sequence, Union
import pickle
import numpy as np
import random

import ray
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset

import src.fed2.train_data as train_data
import src.fed2.model


class Client:
    def __init__(self, model_fn, cid) -> None:
        super().__init__()
        self.dataset_indices = None
        self.model_fn = model_fn
        self.cid = cid
        self.cost = None
        self.last_updated_round = 0
        self.delta_param = 0


def set_parameters(model, parameters) -> torch.nn.Module:
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

    return model


def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def train_fed_proxy(max_epochs, learning_rate, trainloader, net, device, mu=0):
    net.to(device)
    net.train(True)
    first_param = [p.clone().detach() for p in net.parameters()]
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=learning_rate,
        weight_decay=1e-5,
        momentum=0.9, dampening=0.1)

    for _ in range(max_epochs):
        for images, labels in trainloader:
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            # if torch.isnan(images).any() or torch.isinf(images).any():
            #     print('invalid input detected')
            loss = criterion(net(images), labels)
            proxy_term = 0.5 * mu * sum(
                torch.linalg.norm(a - b) ** 2
                for a, b in zip(first_param, net.parameters()))
            loss += proxy_term
            loss.backward()
            nn.utils.clip_grad.clip_grad_norm_(net.parameters(), 1)
            optimizer.step()


def test_model(net: nn.Module, testloader, device=None):
    """Validate the network on the entire test set."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.train(False)
    criterion = torch.nn.CrossEntropyLoss()
    correct, total = 0, 0
    losses = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            # if torch.isnan(images).any() or torch.isinf(images).any():
            #     print('invalid input detected')
            outputs = net(images)
            losses.append(criterion(outputs, labels).item())
            # if np.isnan(losses).any():
            #     print('invalid loss detected')
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    loss_avg = fsum(losses) / len(testloader)
    # if loss_avg > 10 or np.isnan(loss_avg):
    #     print('invalid loss')
    return loss_avg, accuracy


def get_model_fn(model_name: str, num_classes=10) -> tuple[Callable[[], nn.Module], tuple]:
    if model_name.lower() == 'cnn0':
        model_fn = partial(src.fed2.model.CnnModel_0, num_classes=num_classes)
        names = src.fed2.model.CnnModelWeightNames_0
    elif model_name.lower() == 'cnn1':
        model_fn = partial(src.fed2.model.CnnModel_1, num_classes=num_classes)
        names = src.fed2.model.CnnModelWeightNames_1
    else:
        raise ValueError(f'Invalid model name: {model_name}')

    return model_fn, names


def distribute_indices_to_clients(clients, indice_filename):
    with open(indice_filename, 'rb') as fp:
        indice_costs = pickle.load(fp)
    indice = []
    for client, index_cost in zip(clients, indice_costs):
        index, cost = index_cost
        indice.append(index)
        client.dataset_indices = index
        client.cost = cost

    return indice


def check_nan(a):
    if type(a) is np.ndarray:
        if np.isnan(a).any():
            return True
    else:
        k = any([check_nan(b) for b in a])
        if k is True:
            return True
    return False


@ray.remote(num_cpus=3, num_gpus=0.5)
class ClientSimulator:
    def __init__(self) -> None:
        self.param_cache = []
        self.cetral_model_param = None
        self.central_model = None
        self.list_indice = None
        self.train_dataset = None
        # os.environ['CUDA_VISIBLE_DEVICES'] = "0" # Comment out when ray is not in local mode
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        eval('setattr(torch.backends.cudnn, "benchmark", False)')

    def set_indice(self, indice: list):
        self.list_indice = indice

    def init_central_model(self, fn):
        self.central_model = fn()

    def set_global_model(self, param):
        self.cetral_model_param = param

    def set_dataset(self, dataset_name: str, dataset_dir='~/fed2') -> None:
        train_dataset_fn, test_dataset_fn = train_data.get_dataset_fn(
            dataset_name)
        self.train_dataset = train_dataset_fn(dataset_dir)
        self.test_dataset = test_dataset_fn(dataset_dir)

    def client_starts_train(self, client_id, learning_rate, batch_size, epchos, round):
        """Train the network on the training set."""
        seed = 123456 + (client_id * 200) + round
        torch.manual_seed(seed)
        random.seed(seed)

        client_train = Subset(self.train_dataset, self.list_indice[client_id]) # type: ignore
        trainloader = DataLoader(
            client_train, batch_size=batch_size, shuffle=True)
        
        net = set_parameters(
            self.central_model, self.cetral_model_param).to(self.device)
        train_fed_proxy(epchos, learning_rate, trainloader, net, self.device)
        trained_param = get_parameters(net)
        
        return trained_param

    def test_client(self, param=None):
        testloader = DataLoader(self.test_dataset, batch_size=3000)
        if param is not None:
            net = set_parameters(self.central_model, param).to(self.device)
        else:
            net = self.central_model
        avg_loss, acc = test_model(net, testloader, self.device) # type: ignore
        
        return (avg_loss, acc)
