import time
import math
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