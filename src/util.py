import time
import math
import numpy as np
import matplotlib.pyplot as plt

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def plot_value_over_epoch(epochs, values, xlabel, ylabel, title, save_file):
    plt.plot(epochs, values, 'k')
    plt.title(title, fontdict=font)
    plt.xlabel(xlabel, fontdict=font)
    plt.ylabel(ylabel, fontdict=font)
    plt.savefig(save_file)

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        #self.min_validation_loss = np.inf
        self.best_bleu = np.NINF

    def early_stop(self, bleu_score):
        if bleu_score > self.best_bleu:
            self.best_bleu = bleu_score
            self.counter = 0
        elif bleu_score < (self.best_bleu + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    '''
    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss >= (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    '''