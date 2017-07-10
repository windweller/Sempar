import matplotlib.pyplot as plt
import numpy as np
import re


def plot_f1_em(file_path):
    f1s = []
    ems = []
    with open(file_path) as f:
        for line in f:
            # extract valid cost (don't care about training right now)
            if 'F1 score' in line:
                numbers = re.findall(r'\d+\.{0,1}\d*', line)
                f1 = float(numbers[1])
                em = float(numbers[2])
                f1s.append(f1)
                ems.append(em)

    print(ems)
    print(f1s)
    plt.plot(ems, label='exact match')
    plt.plot(f1s, label="f1")

    plt.xlabel('iterations')
    plt.ylabel('score')
    plt.legend(loc=4)

    return f1s, ems

if __name__ == '__main__':
    plot_f1_em("epoch10_log.txt")
    plt.show()