import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP_ONG_PRECOND(nn.Module):
    def __init__(self, out_dim=10, in_channel=1, img_sz=32, hidden_dim=256, dropout=0.):
        print("🐐🐐🐐🐐 Using nn.Module of MLP_ONG_PRECOND")
        super(MLP_ONG_PRECOND, self).__init__()
        self.in_dim = in_channel*img_sz*img_sz
        self.linear = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.Dropout(p=dropout),
            #nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(p=dropout),
            #nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=False),
        )
        self.last = nn.Linear(hidden_dim, out_dim)  # Subject to be replaced dependent on task

    def features(self, x):
        x = self.linear(x.view(-1,self.in_dim))
        return x

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x

class MLP_ONG(nn.Module):
    def __init__(self, out_dim=10, in_channel=1, img_sz=32, hidden_dim=256, dropout=0.):
        super(MLP_ONG, self).__init__()
        self.in_dim = in_channel*img_sz*img_sz
        self.fc1 = nn.Linear(self.in_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc3 = nn.Linear(hidden_dim, out_dim, bias=False)

        self.W = [self.fc1.weight, self.fc2.weight, self.fc3.weight]

    def forward(self, x):
        ## input into model must already be flattened, of shape (batch_size, in_dim)
        if x.shape[1] != self.in_dim:
            # print("⚠️⚠️⚠️ Warning, model was fed in non-flattened data. This is fine during validation, but should not happen during training!!")
            x = x.view(-1, self.in_dim)
        a1 = self.fc1(x)
        h1 = F.relu(a1)
        a2 = self.fc2(h1)
        h2 = F.relu(a2)
        z = self.fc3(h2)

        cache = (a1, h1, a2, h2)

        if torch.is_grad_enabled():
            z.retain_grad()
            for c in cache:
                c.retain_grad()

        return z, cache


class MLP(nn.Module):
    def __init__(self, out_dim=10, in_channel=1, img_sz=32, hidden_dim=256, dropout=0.):
        super(MLP, self).__init__()
        self.in_dim = in_channel*img_sz*img_sz
        self.linear = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.Dropout(p=dropout),
            #nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(p=dropout),
            #nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.last = nn.Linear(hidden_dim, out_dim)  # Subject to be replaced dependent on task

    def features(self, x):
        x = self.linear(x.view(-1,self.in_dim))
        return x

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


class StableMLP(nn.Module):
    # https://github.com/imirzadeh/stable-continual-learning/blob/master/stable_sgd/models.py
    # https://proceedings.neurips.cc/paper/2020/file/518a38cc9a0173d0b2dc088166981cf8-Supplemental.pdf
    def __init__(self, out_dim=10, in_channel=1, img_sz=32, hidden_dim=256, dropout=0.):
        super(StableMLP, self).__init__()
        self.in_dim = in_channel*img_sz*img_sz
        self.linear = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.Dropout(p=dropout),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(p=dropout),
            nn.ReLU(inplace=True),
        )
        self.last = nn.Linear(hidden_dim, out_dim)  # Subject to be replaced dependent on task

    def features(self, x):
        x = self.linear(x.view(-1,self.in_dim))
        return x

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


class ToyMLP(nn.Module) :
    def __init__(self, out_dim=10, in_channel=1, img_sz=32, hidden_dim=256):
        self.in_dim = in_channel * img_sz * img_sz
        self.linear = nn.Sequential()
        self.last = nn.Linear(hidden_dim, out_dim)  # Subject to be replaced dependent on task

    def features(self, x):
        x = self.linear(x.view(-1,self.in_dim))
        return x

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x

def MLP50():
    print("\n Using MLP100 \n")
    return MLP(hidden_dim=50)


def MLP100():
    print("\n Using MLP100 \n")
    return MLP(hidden_dim=100)


def MLP400():
    return MLP(hidden_dim=400)


def MLP1000():
    print("\n Using MLP1000 \n")
    return MLP(hidden_dim=1000)


def MLP2000():
    return MLP(hidden_dim=2000)


def MLP5000():
    return MLP(hidden_dim=5000)

if __name__ == '__main__':
    def count_parameter(model):
        return sum(p.numel() for p in model.parameters())
    
    model = MLP100()
    n_params = count_parameter(model)
    print(f"LeNetC has {n_params} parameters")
