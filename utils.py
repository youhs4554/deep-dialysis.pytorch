import logging
import os
import time

import numpy as np
import pandas as pd
import torch
from lifelines.utils import concordance_index
from torch.utils.data import Subset, DataLoader
import scipy.stats as st

import random


def to_sksurv_format(e, y):
    e = e.astype(bool)
    y = y.astype(float)
    df = pd.DataFrame(np.c_[e, y], columns=['e', 'y'])
    df['e'] = df['e'].astype(bool)
    return pd.DataFrame.to_records(df, index=False)


def seed_all(random_seed=42):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def assign_device(batch, device):
    for k in batch:
        batch[k] = batch[k].to(device)


def c_index(risk_pred, y, e):
    ''' Performs calculating c-index

    :param risk_pred: (np.ndarray or torch.Tensor) model prediction
    :param y: (np.ndarray or torch.Tensor) the times of event e
    :param e: (np.ndarray or torch.Tensor) flag that records whether the event occurs
    :return c_index: the c_index is calculated by (risk_pred, y, e)
    '''
    if not isinstance(y, np.ndarray):
        y = y.detach().cpu().numpy()
    if not isinstance(risk_pred, np.ndarray):
        risk_pred = risk_pred.detach().cpu().numpy()
    if not isinstance(e, np.ndarray):
        e = e.detach().cpu().numpy()
    return concordance_index(y, risk_pred, e)


def adjust_learning_rate(optimizer, epoch, lr, lr_decay_rate):
    ''' Adjusts learning rate according to (epoch, lr and lr_decay_rate)
    :param optimizer: (torch.optim object)
    :param epoch: (int)
    :param lr: (float) the initial learning rate
    :param lr_decay_rate: (float) learning rate decay rate
    :return lr_: (float) updated learning rate
    '''
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr / (1 + epoch * lr_decay_rate)
    return optimizer.param_groups[0]['lr']


class AverageMeter():
    def __init__(self):
        self.reset()

    def update(self, value, count=1):
        self.count += count
        self.sum += value * count

    def reset(self):
        self.sum = 0.0
        self.count = 0

    @property
    def avg(self):
        return self.sum / self.count


@torch.no_grad()
def bootstrap_eval_torch(model, test_ds, device, nb_bootstrap=100):
    model.eval()

    nb_samples = len(test_ds)
    metrics = []

    def resample():
        indices = np.random.choice(range(nb_samples), nb_samples, replace=True)
        return Subset(test_ds, indices)

    for i in range(nb_bootstrap):
        test_ds_res = resample()
        test_loader = DataLoader(test_ds_res, batch_size=nb_samples, shuffle=False)
        for batch in test_loader:
            assign_device(batch, device=device)
            with torch.no_grad():
                out_dict = model(batch['X'])
            cindex = c_index(-out_dict['risk_pred'], batch['y'], batch['e'])
        metrics.append(cindex)

    # Find mean and 95% confidence interval
    mean = np.mean(metrics)
    conf_interval = st.t.interval(0.95, len(metrics) - 1, loc=mean, scale=st.sem(metrics))
    return {
        'mean': mean,
        'confidence_interval': conf_interval
    }


def bootstrap_eval_sksurv(model, test_data, nb_bootstrap=100):
    X_test, e_test, y_test = test_data.X, test_data.e, test_data.y
    Y_test = to_sksurv_format(e_test, y_test)

    metrics = []
    for i in range(nb_bootstrap):
        metrics.append(model.score(X_test, Y_test))

    # Find mean and 95% confidence interval
    mean = np.mean(metrics)
    conf_interval = st.t.interval(0.95, len(metrics) - 1, loc=mean, scale=st.sem(metrics))
    return {
        'mean': mean,
        'confidence_interval': conf_interval
    }

def create_logger(logs_dir):
    ''' Performs creating logger
    :param logs_dir: (String) the path of logs
    :return logger: (logging object)
    '''
    os.makedirs(logs_dir, exist_ok=True)
    # logs settings
    log_file = os.path.join(logs_dir,
                            time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())) + '.log')

    # initialize logger
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    # initialize handler
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    # initialize console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    # builds logger
    logger.addHandler(handler)
    logger.addHandler(console)

    return logger
