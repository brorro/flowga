from random import choice, shuffle, choices, randint, seed

from torch.utils.data import Dataset
from tqdm import tqdm
from pickle import dump
from pathlib import Path
from os import remove

from src.fed2.train_data import *

# Split train data into disjoint set.
# The sets will have at most 5 classes out of 10.


class container:
    def __init__(self, max_classes) -> None:
        self.classes = []
        self.indices = []

    def add_index(self, index, class_int) -> bool:
        if class_int not in self.classes and len(self.classes) > 5:
            return False
        if class_int not in self.classes:
            self.classes.append(class_int)
        self.indices.append(index)

        return True


if __name__ == '__main__':
    seed(2664)

    dataset_name = input(
        "Enter dataset name(fashion_mnist): ") or "fashion_mnist"
    train_fn, _ = get_dataset_fn(dataset_name)
    train_data: Dataset = train_fn('./')

    targets = train_data.targets.numpy() # type: ignore
    zipped = [(x, y) for x, y in enumerate(targets)]
    categorized = [[] for _ in range(len(train_data.classes))] # type: ignore
    for x, y in enumerate(targets):
        categorized[y].append(x)

    # num_clients = int(input("Enther the number of clients(100): ") or 100)
    num_clients = 100
    max_classes = int(input("Enther the maximum number of classes(4): ") or 4)
    # filename = input("Enther save name(dataset.pickle): ") or "dataset.pickle"
    # filename = 'sc2.pickle'
    filename = 'sc1.pickle'
    weights = [20] * 25 + [10] * 50 + [5] * 25

    clients = [container(max_classes) for _ in range(num_clients)]
    with tqdm(total=len(zipped)) as pbar:
        while len(zipped) > 0:
            x, y = zipped.pop()
            while True:
                client = choices(clients, weights, k=1)[0]
                successed = client.add_index(x, y)
                if successed:
                    pbar.update(1)
                    break
    for client in clients:
        shuffle(client.indices)
    indices = [client.indices for client in clients]
    indices = sorted(indices, key=lambda x: len(x), reverse=True)

    results = []
    max_index = len(max(indices, key=lambda x: len(x)))
    min_index = len(min(indices, key=lambda x: len(x)))
    for index in indices:
        #### Random cost / sc2 ######################################################
        # cost = randint(3, 30)
        #############################################################################

        # Imbalanced  / sc 1 ########################################################
        cost = 33 - ((((len(index) - min_index) * 27) / (max_index - min_index)) + 3)
        #############################################################################

        results.append([index, cost])

    ratio = [len(x) / y for x, y in results]
    costs = [y for _, y in results]
    for x in costs:
        print(x)

    path = Path(filename)
    if path.exists():
        remove(str(path))

    with open(filename, 'xb') as fp:
        dump(results, fp)

    print('Done')
