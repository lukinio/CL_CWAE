import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def get_avg_acc(log_file, runs, task):
    regex = r"Task \d+ average acc: (\d+\.\d+)"
    txt = Path(log_file).read_text()
    matches = re.findall(regex, txt)
    avg_acc = np.array(matches).astype(float)
    avg_acc = avg_acc.reshape((runs, task))
    avg_acc = np.mean(avg_acc, axis=0)
    return avg_acc


def get_loses(log_file, runs, task, epoch):
    regex = r"\* TRAIN - Accuracy (\d+\.\d+) Loss (\d+\.\d+)\s*\* VALID - Accuracy (\d+\.\d+) Loss (\d+\.\d+)"
    txt = Path(log_file).read_text()
    matches = re.findall(regex, txt)

    train_acc, train_loss, valid_acc, valid_loss = [], [], [], []
    for m in matches:
        train_acc.append(m[0])
        train_loss.append(m[1])
        valid_acc.append(m[2])
        valid_loss.append(m[3])

    train_acc = np.array(train_acc).astype(float)
    valid_acc = np.array(valid_acc).astype(float)
    train_acc = train_acc.reshape((runs, task, epoch))
    valid_acc = valid_acc.reshape((runs, task, epoch))
    train_acc = np.mean(train_acc, axis=0)
    valid_acc = np.mean(valid_acc, axis=0)

    return train_acc, valid_acc


def plot_learn(path, setting, files, fig_name, runs, task, epoch, title=""):
    for c, file in enumerate(files):
        _, valid_acc = get_loses(path+setting+file+".log", runs, task, epoch)
        X = np.arange(1, valid_acc.shape[0]*valid_acc.shape[1]+1)
        plt.plot(X, valid_acc.reshape(-1), label=file.replace("_", "-"))

    plt.title(title)
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("Accuracy", fontsize=18)
    # plt.xticks(X.astype(int))
    plt.legend(loc="best")
    plt.savefig(f"learn_{setting}_{fig_name}.png")
    plt.show()


def plot_avg_acc(path, setting, files, fig_name, runs, task, title=""):
    for c, file in enumerate(files):
        avg_acc = get_avg_acc(path+setting+file+".log", runs, task)
        X = np.arange(1, avg_acc.shape[0]+1)
        plt.plot(X, avg_acc, label=file.replace("_", "-"))

    plt.title(title, fontsize=36)
    plt.xlabel("Task", fontsize=18)
    plt.ylabel("Accuracy", fontsize=18)
    plt.xticks(X.astype(int))
    plt.legend(loc="lower left")
    plt.savefig(f"avg_acc_{setting}{fig_name}.png")
    plt.show()


if __name__ == '__main__':
    # path = "cnn/split_CIFAR10/"
    # path = "mlp2/split_MNIST/"
    path = "split_CIFAR10_resnet50_baseline/"
    files = ("CW_TaLaR", "CW_TaLaR_enc", "SI", "MAS", "L2", "EWC", "EWC_online")
    files = ("cw_talar_Adam_best", "cw_talar_SDG_best")
    files = ("CW_TaLaR", "SI", "MAS", "L2", "EWC", "EWC_online")
    # files = ("SI", "MAS", "L2", "EWC", "EWC_online")
    setting = "ID_"
    runs, task, epoch = 3, 5, 12

    # plot_learn(path, setting, files, "", runs, task, epoch)
    plot_avg_acc(path, setting, files, "resnet", runs, task)


