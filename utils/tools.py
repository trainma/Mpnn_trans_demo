import numpy as np
import torch


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            10: 5e-5, 15: 3e-5, 30: 1e-6, 50: 8e-7,
            60: 5e-7, 80: 3e-7, 100: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {
            15: 1e-4, 25: 3e-4, 35: 1e-5, 50: 1e-6,
            80: 1e-7
        }

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping():
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.last_score = None

    def __call__(self, val_loss, model, path):
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score:
            self.last_score = self.best_score
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        else:
            print("val loss No decreased")
        # elif score < self.best_score + self.delta:
        #     self.counter += 1
        #     print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
        #     if self.counter >= self.patience:
        #         self.early_stop = True
        # else:
        #     pass
        #     self.best_score = score
        #     self.save_checkpoint(val_loss, model, path)
        #     self.counter = 0

    def save_checkpoint(self, val_loss, model, path):

        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  ')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        print("have save model {}".format(path + '/' + 'checkpoint.pth'))
        self.val_loss_min = val_loss
