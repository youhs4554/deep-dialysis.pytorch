import os
import numpy as np
import models
from datasets import SurvivalDataset, SurvivalDataset2
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import optuna
import joblib

from utils import AverageMeter, c_index, bootstrap_eval_torch, assign_device, seed_all, adjust_learning_rate, \
    to_sksurv_format, bootstrap_eval_sksurv, create_logger
import prettytable as pt
from torch.utils.tensorboard import SummaryWriter


def get_objective(dataset_file, model_class, dataset_class, backend):
    if backend == 'torch':
        # tensorboard logger
        dataset_name = os.path.basename(os.path.dirname(dataset_file))
        log_dir = os.path.join('tb_logs', model_class.__name__, dataset_name)
        train_writer = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
        valid_writer = SummaryWriter(log_dir=os.path.join(log_dir, 'valid'))

    def objective_torch(trial):
        trial_id = f"trial_{study.trials[-1].number}"

        seed_all(42)

        data = dataset_class(dataset_file, is_train=True)
        train_ixs, valid_ixs = train_test_split(np.arange(len(data)),
                                                test_size=0.2,
                                                shuffle=True,
                                                stratify=data.e)
        train_ds, valid_ds = Subset(data, train_ixs), Subset(data, valid_ixs)
        train_loader = DataLoader(train_ds, batch_size=len(train_ds), shuffle=True)
        valid_loader = DataLoader(valid_ds, batch_size=len(valid_ds), shuffle=False)

        if model_class.__name__ == 'DeepSurv':
            model = model_class(data.ndim, trial)
            criterion = models.NegativeLogLikelihood()
        elif model_class.__name__ == 'DynamicDeepSurv':
            model = model_class(data.ndim, data.max_length, trial)
            criterion = models.EventLoss(alpha=trial.suggest_float('criterion__alpha', 0.0, 10.0),
                                         beta=trial.suggest_float('criterion__beta', 0.0, 10.0),
                                         gamma=trial.suggest_float('criterion__gamma', 0.0, 10.0),
                                         )

        base_lr = trial.suggest_float("base_lr", 1e-4, 5e-3)
        lr_decay_rate = trial.suggest_float("lr_decay_rate", 1e-4, 1e-2)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=base_lr)

        model.to(device)
        criterion.to(device)

        loss_meter = AverageMeter()
        score_meter = AverageMeter()
        best_score = 0.0
        best_model = model
        es_count = 0

        for epoch in range(EPOCHS):
            adjust_learning_rate(optimizer, epoch, base_lr, lr_decay_rate)

            # Training
            loss_meter.reset()
            score_meter.reset()
            model.train()
            for batch in train_loader:
                assign_device(batch, device=device)
                out_dict = model(batch.pop('X'))
                loss = criterion(**out_dict, **batch)

                cindex = c_index(-out_dict['risk_pred'], batch['y'], batch['e'])

                # Update meters
                loss_meter.update(loss.item())
                score_meter.update(cindex)

                # Update model
                model.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss = loss_meter.avg
            train_score = score_meter.avg

            # Evaluating
            loss_meter.reset()
            score_meter.reset()
            model.eval()
            for batch in valid_loader:
                assign_device(batch, device=device)
                with torch.no_grad():
                    out_dict = model(batch.pop('X'))
                loss = criterion(**out_dict, **batch)
                cindex = c_index(-out_dict['risk_pred'], batch['y'], batch['e'])

                # Update meters
                loss_meter.update(loss.item())
                score_meter.update(cindex)

            valid_loss = loss_meter.avg
            valid_score = score_meter.avg

            if valid_score > best_score:
                best_score = valid_score
                best_model = model
                es_count = 0
            else:
                es_count += 1
                if es_count >= PATIENCE:
                    print(f"\nEarly Stopping with best_score={best_score:.8f}...")
                    hparams = trial.params
                    hparams['name'] = trial_id
                    train_writer.add_hparams(hparams,
                                          {'valid_cindex': best_score})
                    return best_score

            trial_score = trial.study.best_value if len(trial.study.trials) > 1 else 0.0

            # Save best trial's model only
            if (valid_score > trial_score) and (best_score == valid_score):
                print(f'\nSaving best model with valid_score:{best_score:.8f}')
                torch.save(best_model, MODEL_SAVE_PATH)

            print(
                f'\rEpoch: {epoch}\tLoss: {train_loss:.8f}({valid_loss:.8f})\tc-index: {train_score:.8f}({valid_score:.8f})',
                end='', flush=False)

            train_writer.add_scalar(f"{trial_id}/loss", train_loss, epoch)
            valid_writer.add_scalar(f"{trial_id}/loss", valid_loss, epoch)
            train_writer.add_scalar(f"{trial_id}/c-index", train_score, epoch)
            valid_writer.add_scalar(f"{trial_id}/c-index", valid_score, epoch)

        print()
        return best_score

    def objective_sksurv(trial):
        seed_all(42)

        data = dataset_class(dataset_file, is_train=True)

        X, e, y = data.X, data.e, data.y
        Y = to_sksurv_format(e, y)

        X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y,
                                                              test_size=0.2,
                                                              shuffle=True,
                                                              stratify=e)

        model = model_class(trial)
        model.fit(X_train, Y_train)

        valid_score = model.score(X_valid, Y_valid)

        trial_score = trial.study.best_value if len(trial.study.trials) > 1 else 0.0

        # Save best trial's model only
        if (valid_score > trial_score):
            joblib.dump(model, MODEL_SAVE_PATH)

        return model.score(X_valid, Y_valid)

    if backend == 'torch':
        return objective_torch
    elif backend == 'sksurv':
        return objective_sksurv


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['DeepSurv', 'DynamicDeepSurv', 'CPH', 'RSF'],
                        help='which model to use')
    parser.add_argument('--epochs', type=int, default=500, help='training epochs')
    parser.add_argument('--trials', type=int, default=30, help='optuna trials')
    args = parser.parse_args()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    EPOCHS = args.epochs
    NB_TRIALS = args.trials
    MODEL_DIR = './model_dir'
    PATIENCE = 50
    os.makedirs(MODEL_DIR, exist_ok=True)

    data_info = [
        ('WHAS', './data/whas/whas_train_test.h5'),
        ('SUPPORT', './data/support/support_train_test.h5'),
        ('METABRIC', './data/metabric/metabric_IHC4_clinical_train_test.h5'),
        ('Rotterdam & GBSG', './data/gbsg/gbsg_cancer_train_test.h5')]

    model_class = getattr(models, args.model)
    dataset_class = SurvivalDataset
    model_name = args.model

    if model_name in ['DeepSurv', 'DynamicDeepSurv']:
        backend = 'torch'
        ext = '.pth'
    else:
        backend = 'sksurv'
        ext = '.pkl'

    if model_name == 'DynamicDeepSurv':
        dataset_class = SurvivalDataset2  # dataset for sequence predictions

    logger = create_logger(logs_dir=os.path.join('logs', model_class.__name__))

    headers = []
    values = []

    for name, dataset_file in data_info:
        MODEL_SAVE_PATH = os.path.join(MODEL_DIR, name, model_class.__name__ + ext)
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

        logger.info(f'Running {name}...')
        train_objective = get_objective(dataset_file, model_class, dataset_class, backend)
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(train_objective, n_trials=NB_TRIALS)

        # Bootstrap Test
        if backend == 'torch':
            test_ds = SurvivalDataset(dataset_file, is_train=False)
            data = dataset_class(dataset_file, is_train=True)
            model = torch.load(MODEL_SAVE_PATH)
            model.to(device)

            result = bootstrap_eval_torch(model, test_ds, device=device, nb_bootstrap=100)
        elif backend == 'sksurv':
            model = joblib.load(MODEL_SAVE_PATH)  # model load
            test_data = dataset_class(dataset_file, is_train=False)
            result = bootstrap_eval_sksurv(model, test_data, nb_bootstrap=100)

        logger.info(
            f"[Test]: {result['mean']:.6f} [{result['confidence_interval'][0]:.6f}, {result['confidence_interval'][1]:.6f}] (95% CI)")
        logger.info('')

        headers.append(name)
        values.append(
            f"{result['mean']:.6f} ({result['confidence_interval'][0]:.6f},{result['confidence_interval'][1]:.6f})")

    tb = pt.PrettyTable()
    tb.field_names = headers
    tb.add_row(values)
    logger.info(tb)
