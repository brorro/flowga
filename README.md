# FLOwGA
This is Python code implementation of FLOwGA.

See [FLOwGA](https://ieeexplore.ieee.org/document/10214577) on IEEE open access.

# How to run
## Environment
1. Create conda environment and install packages with following codes.
```
conda install matplotlib tqdm torchvision pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -U "ray"
pip install tbparse tensorboard
```
2. Activate the environment 
3. Run experiments by `python ./run_experiment.py`
## Options
```
usage: run_experiment.py [-h] [--lr lr] [--batch_size batch_size]
                         [--epochs_per_client epochs_per_client]
                         [--dataset_dir dataset_dir] [--writer_dir writer_dir]
                         [--num_genes num_genes] [--threshold threshold]
                         [--repeat repeat] [--seed seed]
                         [--repeat_continue repeat_continue]
                         algorithm round alpha beta dataset_file

positional arguments:
  algorithm             Alogirhtm to run
  round                 FL rounds till terminate
  alpha                 alpha
  beta                  beta
  dataset_file          Name of dataset file

options:
  -h, --help            show this help message and exit
  --lr lr               learning rate
  --batch_size batch_size
                        batch size in client
  --epochs_per_client epochs_per_client
                        epochs in local training
  --dataset_dir dataset_dir
                        Path of dir containing dataset file. This will be used
                        in each clients
  --writer_dir writer_dir
                        Path of dir to write logs and models
  --num_genes num_genes
                        The nubmer of genes in the server when FLOwGA runs
  --threshold threshold
                        Threshold when CMFL runs
  --repeat repeat       The number of runs of the experiment
  --seed seed           seed
  --repeat_continue repeat_continue
                        Continue experiments from run 'continue'th
```
## Example
SGD with a learning rate of 1E-3.

α is set to 0.9, so out of 30 clients, 27 clients will be selected from the chromosome with the greatest fitness.

β was fixed at 0.3, allowing a total of 30 clients to train in one round. , and 3 clients will be selected randomly.

Run FL with FLOwGA in 100 rounds in Scenario 1. Repeat experiments 5 times.
```
python ./run_experiment.py flowga 100 0.9 0.3 'sc1.pickle' --repeat 5
```

# Results
Below are parts of the result.

<img src=acc_sc1.png width=47%>
<img src=acu_sc1.png width=47%>