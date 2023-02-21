import torch
import torch.nn as nn
from lifelines.utils import concordance_index
from torch.autograd import Variable


class DenseBlock(nn.Sequential):
    def __init__(self, in_feats: int, out_feats: int, activation: str, dropout: float, norm: bool, name: str):
        super().__init__()

        def get_name(module):
            return name + '_' + module.__name__

        self.add_module(get_name(nn.Linear), nn.Linear(in_feats, out_feats))
        if norm: self.add_module(get_name(nn.BatchNorm1d), nn.BatchNorm1d(out_feats))
        if dropout: self.add_module(get_name(nn.Dropout), nn.Dropout(dropout))
        activation = getattr(nn, activation)
        self.add_module(get_name(activation), activation())


class DeepSurv(nn.Module):
    def __init__(self, in_feats, trial):
        super().__init__()
        self._build(in_feats, trial)

    def _build(self, in_feats, trial):
        layers = []
        activation = trial.suggest_categorical(f"activation", ["ReLU", "LeakyReLU"])
        norm = trial.suggest_categorical(f"norm", [False, True])
        for i in range(trial.suggest_int("n_layers", 1, 4)):
            name = f"DenseBlock{i}"
            units = trial.suggest_int(f"{name}__units", 32, 256)
            dropout = trial.suggest_float(f"{name}__dropout", 0.0, 0.5)

            layers.append(DenseBlock(in_feats, units, activation, dropout, norm, name=name))

            # update in_feats
            in_feats = units

        self.feature = nn.Sequential(*layers)
        self.fc = nn.Linear(units, 1)

    def get_feature(self, x):
        return self.feature(x)

    def forward(self, x):
        x = self.feature(x)
        out = self.fc(x)
        out = nn.functional.softplus(out)
        return {'risk_pred': out}


class GRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, dropout=dropout if num_layers > 1 else 0.0,
                          batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(x.device)
        out, (hn) = self.gru(x, (h_0))
        out = self.fc(out)
        out = out.squeeze(2)
        out = torch.sigmoid(out)
        return out


class DynamicDeepSurv(nn.Module):
    def __init__(self, in_feats, max_length, trial):
        super().__init__()
        self.static = DeepSurv(in_feats, trial)
        hidden_units = self.static.fc.in_features
        self.dynamic = self._build_model(hidden_units, trial)

        self.max_length = max_length

    def _build_model(self, in_feats, trial):
        return GRU(
            input_size=in_feats,
            output_size=1,
            hidden_size=trial.suggest_int("rnn__units", 32, 256),
            num_layers=trial.suggest_int("rnn__n_layers", 1, 4),
            dropout=trial.suggest_float("rnn__dropout", 0.0, 0.5),
        )

    def forward(self, x):
        feats = self.static.get_feature(x)
        risk_pred = self.static.fc(feats)
        event_seq = self.dynamic(torch.unsqueeze(feats, 1).repeat(1, self.max_length, 1))
        return {'risk_pred': risk_pred, 'event_seq': event_seq}


class NegativeLogLikelihood(nn.Module):
    def __init__(self):
        super(NegativeLogLikelihood, self).__init__()

    def forward(self, risk_pred, y, e):
        mask = torch.ones(y.shape[0], y.shape[0]).to(y.device)
        mask[(y.T - y) > 0] = 0
        log_loss = torch.exp(risk_pred) * mask
        log_loss = torch.sum(log_loss, dim=0) / torch.sum(mask, dim=0)
        log_loss = torch.log(log_loss).reshape(-1, 1)
        neg_log_loss = -torch.sum((risk_pred - log_loss) * e) / torch.sum(e)
        return neg_log_loss


class ConLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, risk_score, y, e):
        device = y.device
        risk_score = risk_score.detach().cpu().numpy()
        y = y.cpu().numpy()
        e = e.cpu().numpy()
        cindex = Variable(
            torch.tensor(concordance_index(y, -risk_score, e))) \
            .to(device)
        cindex.requires_grad_(True)
        return 1.0 - cindex


class EventLoss(nn.Module):
    def __init__(self, alpha, beta):
        super().__init__()
        self.criterion1 = NegativeLogLikelihood()
        self.criterion2 = nn.BCELoss()
        self.criterion3 = ConLoss()

        self.alpha = alpha
        self.beta = beta

    def forward(self, risk_pred, event_seq, y, e, s):
        risk_score = e.squeeze(1) * torch.diag(event_seq[:, s.argmax(1)])

        loss1 = self.criterion1(risk_pred, y, e)  # NegativeLogLikelihood (used in DeepSurv)
        loss2 = self.criterion2(event_seq, s)  # BCE (event sequence prediction)
        loss3 = self.criterion3(risk_score, y, e)  # ConLoss (1.0 - cindex)

        return loss1 + self.alpha * loss2 + self.beta * loss3


def CPH(trial):
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    return CoxnetSurvivalAnalysis(l1_ratio=trial.suggest_float('l1_ratio', 0.5, 1.0),
                                  alpha_min_ratio=0.01)


def RSF(trial):
    from sksurv.ensemble import RandomSurvivalForest
    return RandomSurvivalForest(n_estimators=trial.suggest_int('n_estimators', 100, 1000),
                                max_depth=trial.suggest_int('max_depth', 1, 10),
                                min_samples_split=trial.suggest_int('min_samples_split', 5, 15),
                                min_samples_leaf=trial.suggest_int('min_samples_leaf', 3, 10),
                                n_jobs=-1)
