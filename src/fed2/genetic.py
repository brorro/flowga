import random

class ClientGene:
    def __init__(self, length, max_clients=None, gene=None) -> None:
        if gene is None:
            self.gene = [0] * length
            indice = random.sample(range(len(self.gene)), k=max_clients)
            for i in indice:
                self.gene[i] = 1
        else:
            self.gene = gene
        self.last_updated_round = 0
        self.fitness = 0

    def to_indice(self):
        result = [i for i, x in enumerate(self.gene) if x != 0]
        return result
    