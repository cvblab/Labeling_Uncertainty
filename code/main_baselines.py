import pandas as pd
import os
import numpy as np
import torch
from utils import set_seeds, predict_dataset, evaluate, select_dataframe, learning_curve_plot
from datasets import MILDataset, MILDataGenerator
from losses import CrossEntropyLoss, ShannonEntropy
from MIL_Model import MILModel

train_gpu = torch.cuda.is_available()
set_seeds(42, train_gpu)

for annotator in [7]:

    # Input variables
    dir_images = ['../data/HCUV/', '../data/HUSC']
    dir_results = '../results/'
    dir_dataframe = '../data/'
    #annotator = 2  # From 1 to 10
    classes = ['lm', 'lms', 'df', 'dfs', 'mfc', 'fx', 'cef']
    images_on_ram = True
    #experiment_name = 'confidence_th_08_alphaplu_1_betaplus_01_alphaminus_01_betaminus_1'
    experiment_name = 'confidence_th_09_alphaplu_1_betaplus_01_alphaminus_01_betaminus_1'
    type_labels = 'soft'  # hard, soft
    labeler = 'annotator'  # 'expert', 'annotator'
    confident_teacher = True
    pseudolabeling = False
    experiment_pseudolabels = 'playing_with_confidence_scheduler_lr_scheduler'

    # Hyper-params
    input_shape = (3, 224, 224)
    max_instances = 200
    epochs = 120
    lr = 1e-3
    smoothing = 0.0
    alpha_ce = 1.0
    beta_entropy = 1

    # Prepare folders
    if not os.path.isdir(dir_results + experiment_name + '/'):
        os.mkdir(dir_results + experiment_name + '/')
    if not os.path.isdir(dir_results + experiment_name + '/' + 'Marker_' + str(annotator) + '/'):
        os.mkdir(dir_results + experiment_name + '/' + 'Marker_' + str(annotator) + '/')

    # Load dataframe
    data_frame = pd.read_excel(dir_dataframe + 'dataframe.xls', sheet_name='Marker_' + str(annotator))

    # Dataset partition 
    dataframe_training = pd.read_excel(dir_dataframe + 'training.xls', sheet_name='Marker_' + str(annotator))
    dataframe_training=select_dataframe(data_frame, dataframe_training)

    dataframe_val = pd.read_excel(dir_dataframe + 'validation.xlsx')
    dataframe_val = select_dataframe(data_frame, dataframe_val)

    dataframe_test = pd.read_excel(dir_dataframe + 'test.xlsx')
    dataframe_test = select_dataframe(data_frame, dataframe_test)

    # Prepare datasets
    train_dataset = MILDataset(dir_images, dataframe_training, classes=classes, images_on_ram=images_on_ram)
    val_dataset = MILDataset(dir_images, dataframe_val, classes=classes, images_on_ram=images_on_ram)
    test_dataset = MILDataset(dir_images, dataframe_test, classes=classes, images_on_ram=images_on_ram)

    # Prepare data generators
    train_generator = MILDataGenerator(train_dataset, batch_size=1, shuffle=True, max_instances=max_instances,
                                       labeler=labeler, type_labels=type_labels)
    val_generator = MILDataGenerator(val_dataset, batch_size=1, shuffle=False, max_instances=max_instances,
                                     labeler=labeler, type_labels='hard')
    test_generator = MILDataGenerator(test_dataset, batch_size=1, shuffle=False, max_instances=max_instances,
                                      labeler='expert', type_labels='hard')

    if pseudolabeling:

        teacher = torch.load(dir_results + experiment_pseudolabels + '/' + 'Marker_' + str(annotator) + '/' + labeler + '_network_weights_last.pth')
        # Pseudolabels predictions
        train_generator.indexes = np.arange(len(train_generator.dataset.data_frame))
        _, Yhat_teacher = predict_dataset(train_generator, teacher, train_gpu=train_gpu)
        # replace values in the dataframe
        for i in range(0, Yhat_teacher.shape[0]):
            train_generator.dataset.data_frame[classes].values[i, :] = Yhat_teacher[i, :] * 100
        train_generator.shuffle = True

    # Network architecture
    model = MILModel(input_shape=input_shape, n_classes=len(classes))

    # Set losses
    Lce = CrossEntropyLoss(classes=len(classes), smoothing=smoothing, train_gpu=train_gpu, type_labels=type_labels)
    if abs(beta_entropy) != 0 or confident_teacher:
        Lh = ShannonEntropy()

    # Set optimizer
    opt = torch.optim.SGD(list(model.parameters()), lr=lr)

    # Move to cuda
    if train_gpu:
        model.cuda()
        Lce.cuda()
        if abs(beta_entropy) != 0 or confident_teacher:
            Lh.cuda()

    # Training loop
    history = []
    val_acc_min = 0

    for i_epoch in range(epochs):
        Y_train = []
        Yhat_train = []

        loss_over_all = 0.0
        metric_over_all = 0.0
        if abs(beta_entropy) != 0 or confident_teacher:
            loss_over_all_H = 0.0

        for i_iteration, (X, Y) in enumerate(train_generator):
            ####################
            # --- Training epoch
            model.train()

            '''
            if confident_teacher:

                if np.max(Y) > 0.7:
                    alpha_ce = 1
                    beta_entropy = 0.1
                else:
                    alpha_ce = 0.1
                    beta_entropy = -1
            '''

            if confident_teacher:

                if np.max(Y) > 0.9:
                    alpha_ce = 1
                    beta_entropy = 0.1
                else:
                    alpha_ce = 0.1
                    beta_entropy = -1.

            X = torch.tensor(X).cuda().float()
            Y = torch.tensor(Y).cuda().float()

            # Move to cuda
            if train_gpu:
                X.cuda()
                Y.cuda()

            # Forward network
            logits = model(X)

            # Estimate losses
            ce = Lce(logits, torch.squeeze(Y))
            L = ce * alpha_ce

            if abs(beta_entropy) != 0 or confident_teacher:
                h = Lh(logits)
                L += h * beta_entropy

            # Backward gradients
            L.backward()
            opt.step()
            opt.zero_grad()

            # Track predictions and losses
            Y_train.append(Y.detach().cpu().numpy())
            Yhat_train.append(torch.softmax(logits, -1).detach().cpu().numpy())
            loss_over_all += ce.item()
            if abs(beta_entropy) != 0 or confident_teacher:
                loss_over_all_H += h.item()

            # Display losses and acc per iteration
            info = "[INFO] Epoch {}/{}  -- Step {}/{}: Lce={:.4f}".format(
                    i_epoch + 1, epochs, i_iteration + 1, len(train_generator), ce)
            if abs(beta_entropy) != 0 or confident_teacher:
                info += " || H={:.4f}".format(h.item())
            print(info, end='\r')

        ################
        # --- Epoch end
        model.eval()

        # Validation predictions
        Y_val, Yhat_val = predict_dataset(val_generator, model, train_gpu=train_gpu)
        # Test predictions
        Y_test, Yhat_test = predict_dataset(test_generator, model, train_gpu=train_gpu)

        # Train metrics
        metrics_train = evaluate(np.squeeze(np.argmax(np.concatenate(Y_train), -1)), np.array(Yhat_train))
        loss_training = loss_over_all / len(train_generator)
        if abs(beta_entropy) != 0 or confident_teacher:
            loss_training_H = loss_over_all_H / len(train_generator)
        # Validation metrics
        metrics_val = evaluate(np.argmax(Y_val, -1), Yhat_val)
        loss_val = Lce(torch.tensor(Yhat_val).cuda(), torch.tensor(Y_val).cuda(), preact=False).detach().cpu().numpy()
        # Test metrics
        metrics_test = evaluate(np.argmax(Y_test, -1), Yhat_test)

        # Display losses per epoch
        info = "[INFO] Epoch {}/{}  -- Step {}/{}: Lce={:.4f} accuracy={:.4f} Lce_val={:.4f} accuracy_val={:.4f}".format(
            i_epoch + 1, epochs, len(train_generator), len(train_generator), loss_training, metrics_train["accuracy"],
            loss_val, metrics_val["accuracy"])
        if abs(beta_entropy) != 0 or confident_teacher:
            info += " || H={:.4f}".format(loss_training_H)
        print(info, end='\n')

        # Track learning curves
        history.append([loss_training, loss_val, metrics_train["accuracy"], metrics_train["f1_macro_avg"],
                        metrics_val["accuracy"], metrics_val["f1_macro_avg"], metrics_test["accuracy"],
                        metrics_test["f1_macro_avg"]])

        # Save learning curves
        history_final = pd.DataFrame(history,
                                     columns=['loss_train', 'loss_val', 'metric_train_acc', 'metric_train_f1',
                                              'metric_val_acc', 'metric_val_f1', 'metric_test_acc', 'metric_test_f1'])
        history_final.to_excel(dir_results + experiment_name + '/' + 'Marker_' + str(annotator) + '/' + labeler + '_lc.xlsx')
        learning_curve_plot(history_final,
                            dir_results + experiment_name + '/' + 'Marker_' + str(annotator), labeler + '_lc_scoring')

        # Save model
        if (metrics_val["accuracy"] > val_acc_min) and ((i_epoch + 1) > epochs - 20):
            print('Validation accuracy improved from ' + str(round(val_acc_min, 5)) + ' to ' + str(
                round(metrics_val["accuracy"], 5)) + '  ... saving model')
            torch.save(model,
                       dir_results + experiment_name + '/' + 'Marker_' + str(annotator) + '/' + labeler + '_network_weights_best.pth')
            val_acc_min = metrics_val["accuracy"]

        # Update learning rate during last epochs
        if (i_epoch + 1) > (epochs - 20):
            for g in opt.param_groups:
                g['lr'] = g['lr'] * 0.90

    # Save last model
    torch.save(model, dir_results + experiment_name + '/' + 'Marker_' + str(annotator) + '/' + labeler + '_network_weights_last.pth')