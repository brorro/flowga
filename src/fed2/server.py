from time import sleep
import ray
from src.fed2.client import  ClientSimulator
from tqdm import trange
from datetime import datetime
import random
import numpy as np
from src.fed2.client import Client, get_parameters, \
    get_model_fn, distribute_indices_to_clients
from pickle import dump
from torch.utils.tensorboard import SummaryWriter # type: ignore
from src.fed2.genetic import ClientGene
# import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.axes
from matplotlib.gridspec import GridSpec
from pickle import load
import torch
from pathlib import Path
from src.fed2.train_data import get_dataset_fn


class Server:
    def __init__(self) -> None:
        self.central_model = None
        self.central_model_param = None
        self.model_fn = None
        self.clients = None
        self.indices = None
        self.genes = None
        self.select_client = None
        self.alpha = None

    def run_exp(self, exp_dict):
        # Check dataset existance
        get_dataset_fn(exp_dict['dataset'])[0](exp_dict['dataset_dir'])
        get_dataset_fn(exp_dict['dataset'])[1](exp_dict['dataset_dir'])

        timestamp = datetime.now()
        timestamp = f'_{timestamp.hour:02}{timestamp.minute:02}{timestamp.second:02}'
        # sns.set_theme()
        random.seed(2664 + exp_dict['seed'])
        torch.manual_seed(2664 + exp_dict['seed'])
        writer = SummaryWriter(
            f"{exp_dict['writer_dir']}/run{exp_dict['run_index']:02}/{exp_dict['algorithm']}")
        self.write_dataset_figure(writer, exp_dict['indice_filename'])
        self.model_fn, self.weight_names = get_model_fn(exp_dict['model'])
        self.clients = [Client(self.model_fn, i) for i in range(exp_dict['num_clients'])]
        self.indices = distribute_indices_to_clients(self.clients, exp_dict['indice_filename'])
        self.select_client = self.select_client_to_be_computed_01
        # clients_rayobj = ray.put(self.clients)
        # traindata_fn, testdata_fn = get_dataset_fn(exp_dict['dataset'])
        # traindata_rayobj = ray.put(traindata_fn(exp_dict['dataset_dir']))
        # testdata_rayobj = ray.put(testdata_fn(exp_dict['dataset_dir']))
        self.central_model_param = get_parameters(self.model_fn())
        algorithm = exp_dict['algorithm']
        self.alpha = exp_dict['alpha'] 
        num_clients_per_round = int(exp_dict['beta'] * len(self.clients))
        self.threshold = exp_dict['threshold']
        self.previous_update = self.central_model_param

        if algorithm == 'flowga':
            self.genes = [
                ClientGene(exp_dict['num_clients'], num_clients_per_round) 
                for _ in range(exp_dict['num_genes'])]

        actor_list, actor_pool = self.create_actorpool()
        self.init_simulators(actor_list, exp_dict, self.model_fn)
        self.distribute_central_model(actor_list)
        
        accumulated_cost = 0
        max_fitness = None
        accumulated_joined = [0] * len(self.clients)
        with trange(1, exp_dict['fl_rounds'] + 1, leave=False) as t:
            for i in t:
                # self.threshold = self.threshold / (t ** (1/2))
                lr = exp_dict['lr']
                joined_clients = self.fl_round(
                    i,
                    actor_pool,
                    lr, 
                    exp_dict['batch_size'],
                    exp_dict['epochs_per_client'],
                    num_clients_per_round, algorithm)
                cost = self.sum_eval_cost(joined_clients, algorithm)
                accumulated_cost += cost
                self.distribute_central_model(actor_list)
                actor_pool.submit(lambda a, v: a.test_client.remote(), None)
                loss, acc = actor_pool.get_next_unordered()
                
                if algorithm == 'flowga':
                    self.update_fitness(num_clients_per_round)
                    self.genes = self.reproduce_genes(len(joined_clients))
                    max_fitness = self.update_fitness(num_clients_per_round)
                
                self.write_log(writer, i, loss, cost, 
                    acc, accumulated_cost, joined_clients, 
                    accumulated_joined, algorithm, max_fitness)

                t.set_postfix(loss=loss, acc=acc)
                sleep(1)
        
        save_dir = Path('runs/models/')
        save_dir.mkdir(parents=True, exist_ok=True)
        with (save_dir / (exp_dict['algorithm'] + timestamp + "_model.pt")).open('wb') as fp:
            dump(self.central_model_param, fp)
        with (save_dir / (exp_dict['algorithm'] + timestamp + "_var.txt")).open('wt') as fp:
            print(exp_dict, file=fp)
        for actor in actor_list:
            ray.kill(actor)
    
    def write_log(
        self, writer: SummaryWriter, round, loss, cost, acc, accumulated_cost, 
        joined_clients, accumulated_joined, algorithm, max_fitness):

        weight_names = self.weight_names
        central_model_param = self.central_model_param
        client_length = len(self.clients) # type: ignore
        delta_params = [x.delta_param for x in self.clients] # type: ignore

        write_logs( writer, round, weight_names, central_model_param,
            client_length, delta_params, self.genes, loss, cost, acc, accumulated_cost, 
            joined_clients, accumulated_joined, algorithm, max_fitness)


    def write_dataset_figure(self, writer, filename):
        with open(filename, 'rb') as fd:
            indices_costs = load(fd)

        indices_len = [len(x) for x, _ in indices_costs]
        costs = [y for _, y in indices_costs]
        ratio = [x / y for x, y in zip(indices_len, costs)]

        fig: matplotlib.figure.Figure
        ax: matplotlib.axes.Axes

        fig = plt.figure(tight_layout=True)
        gs = GridSpec(2, 2)

        ax = fig.add_subplot(gs[0, 1])
        ax.bar(x=[i for i in range(len(indices_len))], height=[x for x in indices_len])
        ax.set_ylabel('# Indice')

        ax = fig.add_subplot(gs[0, :-1])
        ax.bar(x=[i for i in range(len(indices_len))], height=[x for x in costs])
        ax.set_ylabel('Cost')

        ax = fig.add_subplot(gs[1, :])
        ax.bar(x=[i for i in range(len(indices_len))], height=[x for x in ratio])
        ax.set_ylabel('Ratio')

        size = fig.get_size_inches()
        size = (size[0] * 2, size[1] * 2)
        fig.set_size_inches(*size)

        writer.add_figure('Dataset', fig)

    def update_fitness(self, limit):
        max_delta = max([client.delta_param for client in self.clients]) # type: ignore
        fitnesses = []
        for gene in self.genes: # type: ignore
            fitness = 0
            for i, x in enumerate(gene.gene):
                if x == 0:
                    continue
                client: Client = self.clients[i] # type: ignore
                fitness += (client.delta_param / max_delta) / client.cost # type: ignore
            gene.fitness = fitness
            fitnesses.append(fitness)
            if sum(gene.gene) > limit:
                gene.fitness = 0
        return max(fitnesses)
            

    def reproduce_genes(self, limit):
        length = len(self.genes[0].gene) # type: ignore
        mutation_prob = 1 / length
        results = []
        
        while len(results) < limit:
            pp1, pp2 = random.sample(self.genes, k=2) # type: ignore
            p1 = max(pp1, pp2, key=lambda x: x.fitness)
            pp1, pp2 = random.sample(self.genes, k=2) # type: ignore
            p2 = max(pp1, pp2, key=lambda x: x.fitness)

            index = random.randint(1, length-1)
            n1_gene = p1.gene[:index] + p2.gene[index:]
            n2_gene = p2.gene[:index] + p1.gene[index:]

            for gene in [n1_gene, n2_gene]:
                for i in range(length):
                    if random.random() < mutation_prob:
                        if gene[i] == 1:
                            gene[i] = 0
                        else:
                            gene[i] = 1
            n1 = ClientGene(length, gene=n1_gene)            
            n2 = ClientGene(length, gene=n2_gene)
            results += [n1, n2]
        return results
            

    def sum_eval_cost(self, joined_clients, algorithm):
        result = 0
        for i in joined_clients:
            result += self.clients[i].cost # type: ignore
        if algorithm == 'cmfl':
            result += sum([self.clients[i].cost for i in range(len(self.clients)) if i not in joined_clients]) / 2

        return result
        
    
    def test_central_model(self, actor_pool):
        actor_pool.submit(lambda a, v: a.test_client.remote(v), self.central_model_param)
        return actor_pool.get_next()

    def init_simulators(self, actor_list, exp_dict, fn):
        ray_objs = []
        for actor in actor_list:
            ray_objs.append(actor.set_dataset.remote(exp_dict['dataset'], exp_dict['dataset_dir']))
            ray_objs.append(actor.set_indice.remote(self.indices))
            ray_objs.append(actor.init_central_model.remote(fn))
        ray.wait(ray_objs)

    def fl_round(self, round, actor_pool, lr, batch_size, epchos, num_clients_per_round, algorithm):        
        if algorithm == "cmfl":
            joined_clients = list(range(len(self.clients)))
            # joined_clients = [0, 1, 2]
        elif algorithm == 'flowga':
            joined_clients = self.select_client(num_clients_per_round) # type: ignore
        else:
            joined_clients = random.choices(range(len(self.clients)), k=num_clients_per_round) # type: ignore
        client_params = list(actor_pool.map(
            lambda a, v: a.client_starts_train.remote(
                v, lr, batch_size, epchos, round), joined_clients))
        
        if algorithm == 'flowga':
            deltas = [sum(np.sum(np.abs(x - y)) for x, y in zip(self.central_model_param, k)) # type: ignore
                        for k in client_params]
            for i, delta in zip(joined_clients, deltas):
                self.clients[i].delta_param = delta # type: ignore
            for i in joined_clients:
                self.clients[i].last_updated_round = round # type: ignore

        if algorithm == "cmfl" and round > 1:
            # self.threshold = self.threshold / (round ** (1 / 30))
            client_params_t, joined_clients_t = self.filter_cmfl(client_params, 
                                             self.threshold, 
                                             self.previous_update,
                                             self.central_model_param)
            if len(joined_clients_t) < 1:
                joined_clients = random.choices(range(len(self.clients)), k=num_clients_per_round) # type: ignore
                client_params = [client_params[x] for x in joined_clients]
            else:
                client_params, joined_clients = client_params_t, joined_clients_t
        elif algorithm == "cmfl" and round == 1:
            joined_clients = random.choices(range(len(self.clients)), k=num_clients_per_round) # type: ignore
            client_params = [client_params[x] for x in joined_clients]
                                             
        
        temp = [sum(j) / len(client_params) for j in zip(*client_params)]

        if algorithm == "cmfl":
            self.previous_update = []
            for x, y in zip(temp, self.central_model_param):
                self.previous_update.append(x - y)

        self.central_model_param = temp

        return joined_clients
    
    def filter_cmfl(self, client_params, threshold, previous_update, previous_model):
        ret1 = []
        ret2 = []
        for i, client_param in enumerate(client_params):
            param = self.get_param_diff(client_param, previous_model)
            pv = self.get_vector_sign_percentage(param, previous_update)
            if pv > threshold:
                ret1.append(client_param)
                ret2.append(i)
        return ret1, ret2
    
    @staticmethod
    def get_param_diff(a, b):
        ret = []
        for x, y in zip(a, b):
            ret.append(x - y)
        return ret


    @staticmethod
    def get_vector_sign_percentage(a, b):
        ret = 0
        length = 0
        for x, y in zip(a, b):
            length += x.size
            ret += np.sum(np.sign(x) == np.sign(y))
        
        return ret / length

    
    def select_client_to_be_computed_01(self, limit):
        # Random selection
        num_clients_from_gene = int(limit * self.alpha)
        indice = []
        for gene in sorted(self.genes, reverse=True, key=lambda x: x.fitness): # type: ignore
            indice = gene.to_indice()
            if not len(indice) < num_clients_from_gene:
                break
        result = random.sample(indice, k=num_clients_from_gene)

        max_delta_index = max(range(len(self.clients)), key=lambda i: self.clients[i].delta_param) # type: ignore
        if max_delta_index not in result:
            result.append(max_delta_index)

        rest = [i for i in range(len(self.clients))] # type: ignore
        while len(result) < limit:
            choosen = random.choice(rest)
            if choosen not in result:
                result.append(choosen)

        return result

    # def select_client_to_be_computed_02(self, limit):
    #     # Uniform selection
    #     num_clients_from_gene = int(limit * self.alpha)
    #     indice = []
    #     for gene in sorted(self.genes, reverse=True, key=lambda x: x.fitness): # type: ignore
    #         indice = gene.to_indice()
    #         if not len(indice) < num_clients_from_gene:
    #             break
    #     result = random.sample(indice, k=num_clients_from_gene)

    #     max_delta_index = max(range(len(self.clients)), key=lambda i: self.clients[i].delta_param) # type: ignore
    #     if max_delta_index not in result:
    #         result.append(max_delta_index)

    #     clients = sorted(range(len(self.clients)), key=lambda x: self.clients[x].last_updated_round) # type: ignore
    #     i = 0
    #     while len(result) < limit:
    #         if clients[i] not in result:
    #             result.append(clients[i])
    #         i += 1

    #     return result
    
    def get_select_method(self, name: str):
        if name == 'gene+random':
            return self.select_client_to_be_computed_01
        elif name == 'gene+uniform':
            return self.select_client_to_be_computed_02
        else:
            raise Exception(f'name error: {name}')


    def distribute_central_model(self, actor_list):
        ray_objs = []
        r = ray.put(self.central_model_param)
        for actor in actor_list:
            obj = actor.set_global_model.remote(r)
            ray_objs.append(obj)
        ray.get(ray_objs)
        

    def create_actorpool(self) -> tuple([list[ClientSimulator], ray.util.ActorPool]): # type: ignore
        num_cluster_gpus = int(ray.available_resources()['GPU']) * 2
        actor_list = [ClientSimulator.remote() for _ in range(num_cluster_gpus)]
        actor_pool = ray.util.ActorPool(actor_list)

        return actor_list, actor_pool


def write_logs(
    writer: SummaryWriter, round, weight_names, central_model_param,
    client_length, delta_params, genes, loss, cost, acc, accumulated_cost, 
    joined_clients, accumulated_joined, algorithm=False, max_fitness=None):
    writer.add_scalar('loss', loss, round)
    writer.add_scalar('cost', cost, round)
    writer.add_scalar('acc', acc, round)
    writer.add_scalar('accumulated_cost', accumulated_cost, round)
    writer.add_text('joined clients', str(sorted(joined_clients)), round)
    for name, value in zip(weight_names, central_model_param):
        writer.add_histogram(name, value, round)
    
    # accumulated joined clients
    fig: matplotlib.figure.Figure
    ax: matplotlib.axes.Axes
    fig, ax = plt.subplots()
    for k in joined_clients:
        accumulated_joined[k] += 1
    ax.bar(x=[i for i in range(client_length)], height=accumulated_joined)
    size = fig.get_size_inches()
    size = [size[0] * 3, size[1]]
    fig.set_size_inches(size) # type: ignore
    writer.add_figure('accumulated joined clients', fig, round)

    # joined clients on current round
    joined = [0] * client_length
    for k in joined_clients:
        joined[k] += 1
    fig, ax = plt.subplots()
    ax.bar(x=[i for i in range(client_length)], height=joined)
    size = fig.get_size_inches()
    size = [size[0] * 3, size[1]]
    fig.set_size_inches(size) # type: ignore
    writer.add_figure('joined clients', fig, round)
    writer.add_scalar('num joined clients', len(joined_clients), round)

    if algorithm == 'flowga':
        writer.add_scalar('max fitness', max_fitness, round)
        best_gene: ClientGene = max(genes, key=lambda x: x.fitness)
        best_gene_clients = [i for i, x in enumerate(best_gene.gene) if x != 0]
        best_gene_txt = f'#{str(sum(best_gene.gene))},  {best_gene.fitness}  ' + f'{best_gene_clients}'
        writer.add_text('best gene(size, fitness, clients)', best_gene_txt, round)

        genes_str = ''
        for k, gene in enumerate(genes):
            indice = [i for i, x in enumerate(gene.gene) if x != 0]
            genes_str += f'{k}({sum(gene.gene)}): {indice} \n\n'
        writer.add_text('All genes', genes_str, round)
        
        fig, ax = plt.subplots()
        ax.bar(
            x=[i for i in range(client_length)],
            height=delta_params)
        size = fig.get_size_inches()
        size = [size[0] * 3, size[1]]
        fig.set_size_inches(size) # type: ignore
        writer.add_figure('delta_params of all', fig, round)

        fig, ax = plt.subplots()
        ax.bar(
            x=[i for i in range(len(joined_clients))],
            height=[delta_params[i] for i in joined_clients])
        size = fig.get_size_inches()
        fig.set_size_inches(size) # type: ignore
        writer.add_figure('delta_params of joined clients', fig, round)

        fig, ax = plt.subplots()
        ax.bar(
            x=[i for i in range(len(best_gene_clients))],
            height=[delta_params[i] for i in best_gene_clients])
        size = fig.get_size_inches()
        fig.set_size_inches(size) # type: ignore
        writer.add_figure('delta_params of the bset gene', fig, round)