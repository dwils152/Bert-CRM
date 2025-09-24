import numpy as np
import sys
import matplotlib.pyplot as plt


def parse_logs(log_file):
    loss, labels, probs = [], [], []
    with open(log_file, 'r') as fin:
        for line in fin:
            if 'Loss' in line:
                loss.append(float(line.split()[-1]))
            elif 'Label' in line:
                labels.append(list(map(int, line.split()[-1].split(','))))
            elif 'Probabilities' in line:
                probs.append(list(map(float, line.split()[-1].split(','))))
    
    return np.array(loss), np.array(labels), np.array(probs)


def plot_loss(train_loss, val_loss):
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('...')
    plt.ylabel('...')
    plt.legend()
    plt.savefig('loss.png', dpi=300)
    
    
    

def main():
    train_log, val_log, test_log = sys.argv[1], sys.argv[2], sys.argv[3]
    train_loss, train_labels, train_probs = parse_logs(train_log)
    val_loss, val_labels, val_probs = parse_logs(val_log)
    test_loss, test_labels, test_probs = parse_logs(test_log)
    
    
    
if __name__ == '__main__':
    main()