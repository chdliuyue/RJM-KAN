import torch

from kan import *
from models_pytorch.utils import MyCrossEntropyLoss, JeffriesMatusitaLoss
from dataPre import datapre
from sklearn.metrics import f1_score


torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.get_default_dtype()


def train_acc():
    return torch.mean((torch.argmax(model(dataset['train_input']), dim=1) == dataset['train_label']).type(dtype))


def test_acc():
    return torch.mean((torch.argmax(model(dataset['test_input']), dim=1) == dataset['test_label']).type(dtype))


def train_f1():
    return f1_score(torch.argmax(model(dataset['train_input']), dim=1).cpu().numpy(),
                    dataset['train_label'].cpu().numpy(), average='macro')


def test_f1():
    return f1_score(torch.argmax(model(dataset['test_input']), dim=1).cpu().numpy(),
                    dataset['test_label'].cpu().numpy(), average='macro')


d = 20
dataset = datapre(d)
train_Loss, test_Loss, train_Acc, test_Acc, train_F1, test_F1 = [], [], [], [], [], []
for i in range(10):
    model = KAN(width=[d, 10, 3], grid=5, k=3, seed=i, device=device)
    results = model.fit(dataset, opt="LBFGS", steps=50, metrics=(train_acc, test_acc, train_f1, test_f1),
                        # loss_fn=torch.nn.CrossEntropyLoss())
                        loss_fn=JeffriesMatusitaLoss())
    train_Loss.append(min(results['train_loss']))
    test_Loss.append(min(results['test_loss']))
    train_Acc.append(max(results['train_acc']))
    test_Acc.append(max(results['test_acc']))
    train_F1.append(max(results['train_f1']))
    test_F1.append(max(results['test_f1']))

print('| train Loss: {:.4f}, std: {:.4f} | '.format(np.mean(train_Loss), np.std(train_Loss)))
print('| test Loss: {:.4f}, std: {:.4f} | '.format(np.mean(test_Loss), np.std(test_Loss)))
print('| train ACC: {:.4f}, std: {:.4f} | '.format(np.mean(train_Acc), np.std(train_Acc)))
print('| test ACC: {:.4f}, std: {:.4f} | '.format(np.mean(test_Acc), np.std(test_Acc)))
print('| train F1: {:.4f}, std: {:.4f} | '.format(np.mean(train_F1), np.std(train_F1)))
print('| test F1: {:.4f}, std: {:.4f} | '.format(np.mean(test_F1), np.std(test_F1)))

model.prune()
model.plot(beta=10, scale=1)
lib = ['x', 'x^2', 'x^3', 'x^4', 'exp', 'log', 'sqrt', 'tanh', 'sin', 'abs']
model.auto_symbolic(lib=lib)

formula1, formula2, formula3, *rest = model.symbolic_formula()[0]
print(ex_round(formula1, 2))
print(ex_round(formula2, 2))
print(ex_round(formula3, 2))




