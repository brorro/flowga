from tbparse import SummaryReader
from pathlib import Path
from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np


def get_mean(paths):
    dfs = []
    for path in paths:
        reader = SummaryReader(path)
        df = reader.scalars
        dfs.append(df)

    value = [x[x['tag'] == 'acc'].value.values for x in dfs]
    value = np.vstack(value)
    mean_acc = np.mean(value, axis=0)

    value = [x[x['tag'] == 'accumulated_cost'].value.values for x in dfs]
    value = np.vstack(value)
    mean_accumulated_cost = np.mean(value, axis=0)

    value = [x[x['tag'] == 'num joined clients'].value.values for x in dfs]
    value = np.vstack(value)
    mean_num_joined = np.mean(value, axis=0)

    value = [x[x['tag'] == 'cost'].value.values for x in dfs]
    value = np.vstack(value)
    mean_cost = np.mean(value, axis=0)

    value = [x[x['tag'] == 'loss'].value.values for x in dfs]
    value = np.vstack(value)
    mean_loss = np.mean(value, axis=0)

    value = [x[x['tag'] == 'max fitness'].value.values for x in dfs]
    if len(value[0]) > 0:
        value = np.vstack(value)
        mean_max_fitness = np.mean(value, axis=0)
    else:
        mean_max_fitness = [-100] * len(mean_loss)


    return mean_acc, mean_accumulated_cost, mean_cost, mean_loss, mean_max_fitness, mean_num_joined


def get_log_groups(path: Path, key: str):
    log_events = [str(x) for x in path.glob('**/events.out*')]

    log_groups = {}
    group = [x for x in log_events if key in x]
    if group:
        log_groups[key] = group
    
    return log_groups


def wrtie_mean_logs(path, algo):
    path = Path(path)
    if not path.exists():
        raise Exception()
    log_groups = get_log_groups(path, algo)
    for key, value in log_groups.items():
        writer = SummaryWriter(path / ('mean/' + key))
        mean_acc, mean_accumulated_cost, mean_cost, mean_loss, mean_max_fitness, mean_num_joined = get_mean(value)
        for i, (acc, acc_cost, cost, loss, max_fitness, num_joined) in enumerate(
            zip(mean_acc, mean_accumulated_cost, mean_cost, mean_loss, mean_max_fitness, mean_num_joined)):
            writer.add_scalar('acc', acc, i + 1)
            writer.add_scalar('accumulated_cost', acc_cost, i + 1)
            writer.add_scalar('cost', cost, i + 1)
            writer.add_scalar('num joined clients', num_joined, i + 1)
            writer.add_scalar('loss', loss, i + 1)
            if max_fitness > -10:
                writer.add_scalar('max fitness', max_fitness, i + 1)
