import os
import numpy as np

from models import DeepSurv, DynamicDeepSurv, NegativeLogLikelihood, EventLoss
from datasets import SurvivalDataset, SurvivalDataset2
import torch
from torch.utils.data import DataLoader, Subset, Sampler
from sklearn.model_selection import train_test_split
import optuna

from utils import AverageMeter, c_index, bootstrap_eval, assign_device, seed_all, adjust_learning_rate


class StratifiedSampler(Sampler):
    """Stratified Sampling
    Provides equal representation of target classes in each batch
    """

    def __init__(self, class_vector, batch_size):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        self.n_splits = int(class_vector.size(0) / batch_size)
        self.class_vector = class_vector

    def gen_sample_array(self):
        try:
            from sklearn.model_selection import StratifiedShuffleSplit
        except:
            print("Need scikit-learn for this functionality")
        import numpy as np

        s = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.7)
        X = torch.randn(self.class_vector.size(0), 2).numpy()
        y = self.class_vector.numpy()
        s.get_n_splits(X, y)

        train_index, test_index = next(s.split(X, y))
        return np.hstack([train_index, test_index])

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)


def get_objective(dataset_file, model_class, dataset_class):
    def objective(trial):

        seed_all(42)

        data = dataset_class(dataset_file, is_train=True)
        train_ixs, valid_ixs = train_test_split(np.arange(len(data)),
                                                test_size=0.2,
                                                shuffle=True,
                                                stratify=data.e)
        train_ds, valid_ds = Subset(data, train_ixs), Subset(data, valid_ixs)
        train_loader = DataLoader(train_ds, batch_size=32,
                                  sampler=StratifiedSampler(
                                      torch.from_numpy(data.e[train_ixs].squeeze()), batch_size=32), drop_last=True)
        valid_loader = DataLoader(valid_ds, batch_size=32, shuffle=False)

        if model_class.__name__ == 'DeepSurv':
            model = model_class(data.ndim, trial)
            criterion = NegativeLogLikelihood()
        elif model_class.__name__ == 'DynamicDeepSurv':
            model = model_class(data.ndim, data.max_length, trial)
            criterion = EventLoss(alpha=trial.suggest_float('criterion__alpha', 0.0, 10.0),
                                  beta=trial.suggest_float('criterion__beta', 0.0, 10.0))

        base_lr = trial.suggest_float("base_lr", 1e-4, 5e-3)
        lr_decay_rate = trial.suggest_float("lr_decay_rate", 1e-4, 1e-2)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=base_lr)

        model.to(device)
        criterion.to(device)

        loss_meter = AverageMeter()
        score_meter = AverageMeter()
        best_score = 0.0
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
                es_count = 0
            else:
                es_count += 1
                if es_count >= PATIENCE:
                    print(f"\nEarly Stopping with best_score={best_score:.8f}...")
                    return best_score

            # Save best trial's model only
            if len(trial.study.trials) > 1:
                if valid_score > trial.study.best_value:
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch}, MODEL_SAVE_PATH
                    )

            print(
                f'\rEpoch: {epoch}\tLoss: {train_loss:.8f}({valid_loss:.8f})\tc-index: {train_score:.8f}({valid_score:.8f})',
                end='', flush=False)

        print()
        return best_score

    return objective


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    EPOCHS = 500
    NB_TRIALS = 30
    MODEL_DIR = './model_dir'
    PATIENCE = 50
    os.makedirs(MODEL_DIR, exist_ok=True)

    # TODO: 조건문으로 경우의 수 나누기
    dataset_file = "./data/metabric/metabric_IHC4_clinical_train_test.h5"
    model_class = DynamicDeepSurv
    dataset_class = SurvivalDataset2

    # TODO: CPH, RSF 추가 (get_objective() 함수에 backbone 변수 추가해서 경우의 수 나누는 방식)
    # TODO: MLFlow tracking
    # TODO: Logger 추가

    MODEL_SAVE_PATH = os.path.join(MODEL_DIR, model_class.__name__ + '.pth')

    train_objective = get_objective(dataset_file, model_class, dataset_class)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(train_objective, n_trials=NB_TRIALS)

    # Load weights and Bootstrap Test
    test_ds = SurvivalDataset(dataset_file, is_train=False)
    data = dataset_class(dataset_file, is_train=True)
    if model_class.__name__ == 'DeepSurv':
        model = model_class(data.ndim, study.best_trial)
    elif model_class.__name__ == 'DynamicDeepSurv':
        model = model_class(data.ndim, data.max_length, study.best_trial)

    model.load_state_dict(torch.load(MODEL_SAVE_PATH)['model'])
    model.to(device)

    result = bootstrap_eval(model, test_ds, device=device, nb_bootstrap=100)
    print()
    print(
        f"\r[Test]: {result['mean']:.8f} [{result['confidence_interval'][0]:.8f}, {result['confidence_interval'][1]:.8f}] (95% CI)")
