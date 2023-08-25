import ray
from src.fed2.server import Server
from time import sleep
from pathlib import Path
from shutil import rmtree, move
from os import remove

from src.fed2.stats import wrtie_mean_logs
from tqdm import trange

import argparse


def del_files(start_path):
    p = Path(start_path)
    for pp in p.iterdir():
        if pp.is_dir() is True:
            rmtree(pp)
        else:
            remove(pp)


def move_runs(start_prefix):
    prefix = Path(start_prefix)
    if prefix.exists() and len(list(prefix.iterdir())) > 0:
        for i in range(999):
            i_str = f"{i:03}"
            dst = Path(f"runs_backup/{i_str}/")
            if not dst.exists():
                dst.mkdir(parents=True)
                break
        move(start_prefix, dst)  # type: ignore


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "algo",
        metavar="algorithm",
        help="Alogirhtm to run",
        choices=["flowga", "cmfl", "fedavg"],
    )
    parser.add_argument(
        "round", metavar="round", help="FL rounds till terminate", type=int
    )
    parser.add_argument("alpha", metavar="alpha", help="alpha", type=float)
    parser.add_argument("beta", metavar="beta", help="beta", type=float)
    parser.add_argument(
        "--lr", metavar="lr", help="learning rate", default=0.001, type=float
    )
    parser.add_argument(
        "--batch_size",
        metavar="batch_size",
        help="batch size in client",
        default=30,
        type=int,
    )
    parser.add_argument(
        "--epochs_per_client",
        metavar="epochs_per_client",
        help="epochs in local training",
        default=3,
        type=int,
    )
    parser.add_argument(
        "dataset_file",
        metavar="dataset_file",
        help="Name of dataset file",
        default="fm_biased_RandomCostRaito.pickle",
    )
    parser.add_argument(
        "--dataset_dir",
        metavar="dataset_dir",
        help="Path of dir containing dataset file. This will be used in each clients",
        default="./",
    )
    parser.add_argument(
        "--writer_dir",
        metavar="writer_dir",
        help="Path of dir to write logs and models",
        default="./runs",
    )
    parser.add_argument(
        "--num_genes",
        metavar="num_genes",
        help="The nubmer of genes in the server when FLOwGA runs",
        default=100,
        type=int,
    )
    parser.add_argument(
        "--threshold",
        metavar="threshold",
        help="Threshold when CMFL runs",
        default=0.65,
        type=float,
    )
    parser.add_argument(
        "--repeat",
        metavar="repeat",
        help="The number of runs of the experiment",
        default=10,
        type=int,
    )
    parser.add_argument("--seed", metavar="seed", help="seed", default=5, type=int)
    parser.add_argument(
        "--repeat_continue",
        metavar="repeat_continue",
        help="Continue experiments from run 'continue'th",
        default=0,
        type=int,
    )

    args = parser.parse_args()

    default = {
        "lr": 0.001,
        "dataset": "fashion_mnist",
        "model": "cnn1",
        "num_clients": 100,
        "indice_filename": "sc1.pickle",
        "fl_rounds": 100,
        "beta": 0.3,
        "batch_size": 30,
        "epochs_per_client": 3,
        "dataset_dir": "~/fed2",
        "num_genes": 100,
        "alpha": 0.7,
        "algorithm": "cmfl",
        "writer_dir": "./runs",
        "seed": 5,
        "repeat": 10,
        "continue": 0,
        "threshold": 0.65,
    }
    default["algorithm"] = args.algo
    default["fl_rounds"] = args.round
    default["alpha"] = args.alpha
    default["beta"] = args.beta
    default["lr"] = args.lr
    default["batch_size"] = args.batch_size
    default["epochs_per_client"] = args.epochs_per_client
    default["indice_filename"] = args.dataset_file
    default["dataset_dir"] = args.dataset_dir
    default["num_genes"] = args.num_genes
    default["repeat"] = args.repeat
    default["continue"] = args.repeat_continue
    default["seed"] = args.seed
    default["threshold"] = args.threshold

    ray.init()
    for i in trange(default["continue"], default["repeat"]):
        exp_dict = default
        exp_dict["run_index"] = i
        server = Server()
        server.run_exp(exp_dict)
        del server
        sleep(5)
    wrtie_mean_logs(args.writer_dir, args.algo)
