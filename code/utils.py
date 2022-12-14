import numpy as np
import pandas as pd
import random
import torch
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from matplotlib import pyplot as plt


def select_dataframe(data_frame, dataframe):

    values=dataframe.to_numpy()
    dataframe = data_frame.loc[data_frame['Name_Image'].isin(values.squeeze(axis=-1))]
    return dataframe


def set_seeds(seed_value, use_cuda):
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False


def predict_dataset(generator, model, train_gpu=True):
    model.eval()
    print('[PREDICTING]: at bag level...')

    # Loop over training dataset
    Y_dataset = []
    Yhat_dataset = []
    for i, (X, Y) in enumerate(generator):
        print(str(i+1) + '/' + str(len(generator)), end='\r')

        X = torch.tensor(X).cuda().float()
        Y = torch.tensor(Y).cuda().float()

        # Move to cuda
        if train_gpu:
            X.cuda()
            Y.cuda()

        # Forward
        logits = model(X)

        # Get probabilities
        Yhat = torch.softmax(logits, dim=-1)

        Y_dataset.append(Y.detach().cpu().numpy())
        Yhat_dataset.append(Yhat.detach().cpu().numpy())

    # Output predictions
    Yhat_dataset = np.array(Yhat_dataset)
    Y_dataset = np.squeeze(np.array(Y_dataset))

    return Y_dataset, Yhat_dataset


def evaluate(refs, preds):

    accuracy = accuracy_score(refs, np.argmax(preds, -1))
    f1 = f1_score(refs, np.argmax(preds, -1), average=None)
    f1_micro_avg = np.mean(f1)
    auc = roc_auc_score(refs, preds, average='macro', multi_class='ovo', labels=np.arange(0, preds.shape[-1]))

    return {'accuracy': accuracy,
            'f1_macro_avg': f1_micro_avg, 'f1': f1,
            'auc_macro_avg': auc}


def learning_curve_plot(history, dir_out, name_out):
    plt.figure()
    plt.subplot(211)
    plt.plot(history['metric_train_acc'].values)
    plt.plot(history['metric_val_acc'].values)
    plt.axis([0, history['loss_train'].values.shape[0] - 1, 0, 1])
    plt.legend(['acc', 'val_acc'], loc='upper right')
    plt.title('learning-curve')
    plt.ylabel('accuracy')
    plt.subplot(212)
    plt.plot(history['loss_train'].values)
    plt.plot(history['loss_val'].values)
    plt.axis([0, history['loss_train'].values.shape[0] - 1, 0, np.max(history['loss_val'].values)])
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(dir_out + '/' + name_out)
    plt.close()


def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """
    pretty print for confusion matrixes
    https://gist.github.com/zachguo/10296432
    """
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth

    fst_empty_cell = (columnwidth - 3) // 2 * " " + "t/p" + (columnwidth - 3) // 2 * " "

    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    # Print header
    print("    " + fst_empty_cell, end=" ")

    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")

    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{}d".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()
