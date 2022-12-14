import pandas as pd
import matplotlib.pyplot as plt

res1 = pd.read_excel('../results/Soft_non_experts/Marker_2/annotator_lc.xlsx')
res2 = pd.read_excel('../results/playing_with_confidence_scheduler_lr_scheduler/Marker_2/annotator_lc.xlsx')

plt.plot(res1['metric_test_acc'])
plt.plot(res2['metric_test_acc'])
plt.show()


res1 = pd.read_excel('../results/Soft_non_experts/Marker_1/annotator_lc.xlsx')
res2 = pd.read_excel('../results/playing_with_confidence_scheduler_lr_scheduler/Marker_1/annotator_lc.xlsx')

plt.plot(res1['metric_test_acc'])
plt.plot(res2['metric_test_acc'])
plt.show()

####

import pandas as pd
from utils import evaluate, select_dataframe
import numpy as np

dir_dataframe = '../data/'
classes = ['lm', 'lms', 'df', 'dfs', 'mfc', 'fx', 'cef']
annotator = 1  # From 1 to 10

# Load dataframe
data_frame = pd.read_excel(dir_dataframe + 'dataframe.xls', sheet_name='Marker_' + str(annotator))
dataframe_test = pd.read_excel(dir_dataframe + 'training.xls')
dataframe_test = select_dataframe(data_frame, dataframe_test)

Y_expert = dataframe_test['GT']
Y_annotator = dataframe_test[classes].values/100
max_confidences = np.max(Y_annotator, 1)
th = 0.7

idx = max_confidences > th

metrics_confident = evaluate(Y_expert[idx], Y_annotator[idx, :])

acc_c = metrics_confident['accuracy']
f1_c = metrics_confident['f1_macro_avg']

metrics_non_confident = evaluate(Y_expert[(np.logical_not(idx))], Y_annotator[(np.logical_not(idx)), :])

acc_nc = metrics_non_confident['accuracy']
f1_nc = metrics_non_confident['f1_macro_avg']

####

import pandas as pd
from utils import evaluate, select_dataframe
import numpy as np

dir_dataframe = '../data/'
classes = ['lm', 'lms', 'df', 'dfs', 'mfc', 'fx', 'cef']
annotator = 10  # From 1 to 10

# Load dataframe
data_frame = pd.read_excel(dir_dataframe + 'dataframe.xls', sheet_name='Marker_' + str(annotator))
dataframe_test = pd.read_excel(dir_dataframe + 'training.xls')
dataframe_test = select_dataframe(data_frame, dataframe_test)

Y_expert = dataframe_test['GT']
Y_annotator = dataframe_test[classes].values/100
confidence = len(np.argwhere(np.max(Y_annotator, -1) == 1)) / Y_annotator.shape[0]

metrics = evaluate(Y_expert, Y_annotator)

#### BAR PLOTS - non-experts and smoothing

import scipy.io
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')

hard_labels = [0.277, 0.288, 0.386, 0.309, 0.448, 0.298, 0.348, 0.259, 0.280, 0.353]
hard_labels_ls = [0.2534, 0.3480, 0.2333, 0.2997, 0.3406, 0.3668, 0.3335, 0.4264, 0.2453, 0.4565]
hard_labels_h = [0.2828, 0.3857, 0.2764, 0.3136, 0.4046, 0.3721, 0.3057, 0.3389, 0.3278, 0.4740]
soft_labels = [0.295, 0.424, 0.401, 0.355, 0.460, 0.304, 0.427, 0.270, 0.323, 0.3902]
leg = ['HL', 'LS', r"$H^{+}$", 'SL']
labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

ind = np.arange(len(hard_labels)/2)
width = 0.2

fig, (ax1, ax2) = plt.subplots(2, 1, sharey=True, figsize=(5, 4))

ax1.bar(ind - 1.5*(width), hard_labels[:(int(len(hard_labels)/2))], width, color='b', alpha=0.6)
ax1.bar(ind - width/2, hard_labels_ls[:(int(len(hard_labels)/2))], width, color='g', alpha=0.6)
ax1.bar(ind + width/2, hard_labels_h[:(int(len(hard_labels)/2))], width, color='y', alpha=0.6)
ax1.bar(ind + 1.5*(width), soft_labels[:(int(len(hard_labels)/2))], width, color='r', alpha=0.6)
ax1.set_xticks(ticks=ind)
ax1.set_xticklabels(labels[:(int(len(hard_labels)/2))])

ax2.bar(ind - 1.5*(width), hard_labels[(int(len(hard_labels)/2)):], width, color='b', alpha=0.6)
ax2.bar(ind - width/2, hard_labels_ls[(int(len(hard_labels)/2)):], width, color='g', alpha=0.6)
ax2.bar(ind + width/2, hard_labels_h[(int(len(hard_labels)/2)):], width, color='y', alpha=0.6)
ax2.bar(ind + 1.5*(width), soft_labels[(int(len(hard_labels)/2)):], width, color='r', alpha=0.6)
ax2.set_xticks(ticks=ind)
ax2.set_xticklabels(labels[(int(len(hard_labels)/2)):])

ax1.legend(leg, bbox_to_anchor=(1, 1), loc=4, frameon=False, fontsize=10,
           ncol=4)

plt.minorticks_on()
ax2.set_xlabel("Annotator", fontsize=10)
ax1.set_ylabel("F1-score", fontsize=10)
ax2.set_ylabel("F1-score", fontsize=10)
ax2.set_yticks([0, 0.2, 0.3, 0.4, 0.5])
ax2.set_ylim([0, 0.5])


plt.savefig('artificial_labeled_calibration.png', dpi=200, format='png', bbox_inches='tight')


## Plot tau value hiperparameter validation

import scipy.io
import matplotlib.pyplot as plt
import numpy as np

labelers = ['Annotator 1', 'Annotator 2', 'Annotator 7', 'Annotator 10']

fig = plt.figure(figsize=(5, 3))

plt.style.use('ggplot')

f1_lab1 = [0.362, 0.362, 0.3238, 0.3238, 0.408, 0.408, 0.224, 0.124]
tau_lab1 = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

f1_lab10 = [0.395, 0.395, 0.3816, 0.381, 0.398, 0.384, 0.308, 0.253]
tau_lab10 = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

f1_lab2 = [0.3628, 0.3628, 0.3819, 0.3819, 0.503, 0.489, 0.3260, 0.2850, ]
tau_lab2 = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

f1_lab7 = [0.4217, 0.4217, 0.3910, 0.3910, 0.4444, 0.4609, 0.4444, 0.4001]
tau_lab7 = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

plt.plot(tau_lab1, f1_lab1, '-', color='royalblue')
plt.plot(tau_lab2, f1_lab2, '-', color='lightcoral')
plt.plot(tau_lab7, f1_lab7, '-', color='wheat')
plt.plot(tau_lab10, f1_lab10, '-', color='green')

plt.legend(labelers, loc=0, frameon=True, fontsize=10,
           ncol=1)

plt.xticks(ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
           labels=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
plt.yticks(ticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5], labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5])

plt.ylabel("F1-score", fontsize=12)
plt.xlabel(r"$\tau$", fontsize=12)
plt.axis([-0.05, 1.05, 0.05, 0.55])

plt.plot(tau_lab1, f1_lab1, '*', markersize=12, color='navy')
plt.plot(tau_lab2, f1_lab2, '*', markersize=12, color='indianred')
plt.plot(tau_lab7, f1_lab7, '*', markersize=12, color='orange')
plt.plot(tau_lab10, f1_lab10, '*', markersize=12, color='darkgreen')

plt.savefig('ablation_tau.png', dpi=200, format='png', bbox_inches='tight')

plt.show()

## Confusion matrixes expert vd. annotator

import pandas as pd
from utils import evaluate, select_dataframe
import numpy as np
import seaborn as sn
from sklearn.metrics import plot_confusion_matrix, confusion_matrix
import matplotlib.pyplot as plt

classes = ['lm', 'lms', 'df', 'dfs', 'mfc', 'fx', 'cef']

CMs = []
for annotator in np.arange(1, 11):

    dir_dataframe = '../data/'
    classes = ['lm', 'lms', 'df', 'dfs', 'mfc', 'fx', 'cef']

    # Load dataframe
    data_frame = pd.read_excel(dir_dataframe + 'dataframe.xls', sheet_name='Marker_' + str(annotator))
    dataframe_test = pd.read_excel(dir_dataframe + 'test.xlsx')
    dataframe_test = select_dataframe(data_frame, dataframe_test)

    Y_expert = dataframe_test['GT']
    Y_annotator = np.argmax(dataframe_test[classes].values/100, 1)
    cm = confusion_matrix(Y_expert, Y_annotator, labels=np.arange(0, len(classes)))

    CMs.append(cm)


CM_avg = np.mean(np.array(CMs), 0)
CM_avg_norm = np.round(CM_avg/CM_avg.sum(1)*100)

plt.figure(figsize = (10, 7))
g = sn.heatmap(CM_avg_norm, annot=True, vmin=0, vmax=100, cmap="YlGnBu",
           )
sn.set(font_scale=2)

g.set_yticklabels(classes)
g.set_xticklabels(classes)


plt.ylabel('Expert Labels')
plt.xlabel('Predicted')

plt.savefig('labels' + '_cm.png', dpi=200, format='png', bbox_inches='tight')
plt.show()


## Average confidence of non-expert labellers for true classes

import pandas as pd
from utils import evaluate, select_dataframe
import numpy as np
import seaborn as sn
from sklearn.metrics import plot_confusion_matrix, confusion_matrix
import matplotlib.pyplot as plt

classes = ['lm', 'lms', 'df', 'dfs', 'mfc', 'fx', 'cef']

p_all = []
percentage_samples_above_th = []
for annotator in np.arange(1, 11):

    dir_dataframe = '../data/'
    classes = ['lm', 'lms', 'df', 'dfs', 'mfc', 'fx', 'cef']

    # Load dataframe
    data_frame = pd.read_excel(dir_dataframe + 'dataframe.xls', sheet_name='Marker_' + str(annotator))
    dataframe_test = pd.read_excel(dir_dataframe + 'training.xls')
    dataframe_test = select_dataframe(data_frame, dataframe_test)

    Y_expert = np.array(dataframe_test['GT'])
    Y_annotator = dataframe_test[classes].values/100

    p_classes = []
    percentage_samples = []
    for i in np.arange(0, len(classes)):
        idx = np.argwhere(Y_expert==i)

        p = Y_annotator[idx, i]
        p_classes.append(np.mean(p[p>0]))
        #percentage_samples.append(np.argwhere(p>0.7).__len__()/p.__len__())
        percentage_samples.append(np.argwhere(Y_annotator.max(1) > 0.7).__len__() / Y_annotator.max(1).__len__())

    p_all.append(np.array(p_classes))
    percentage_samples_above_th.append(np.array(percentage_samples))

p_all = np.array(p_all)
percentage_samples_above_th = np.array(percentage_samples_above_th)
avg = np.nanmean(p_all, 0)


## Bar plots to explain variability on the results

import pandas as pd
from utils import evaluate, select_dataframe
import numpy as np
import seaborn as sn
from sklearn.metrics import plot_confusion_matrix, confusion_matrix
import matplotlib.pyplot as plt

# 1. Accuracy on training dataset from non-experts

f1_train = [0.4581, 0.4589, 0.3456, 0.5342, 0.4745, 0.4701, 0.4591, 0.5107, 0.3198, 0.4790]

ind = np.arange(len(f1_train))
width = 0.4

fig = plt.figure(figsize=(5, 1))

plt.bar(ind, f1_train, width, color='peru', alpha=0.6)
plt.xticks(ticks=ind, labels=np.arange(1, 11))
plt.yticks(ticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

plt.minorticks_on()
plt.xlabel("Annotator", fontsize=10)
plt.ylabel("F1-score", fontsize=10)
plt.axis([-0.5, 9.5, 0.00, 0.6])

plt.plot(np.arange(-1, 12), np.ones(len(np.arange(-1, 12)))*0.4510, 'r', linestyle='dashed')

plt.savefig('metrics_train_set' + '.png', dpi=200, format='png', bbox_inches='tight')

plt.show()


# 2. Percentage of samples above the threshold

percentage_above_th = [0.7027, 0.7522, 0.6311, 0.8655, 0.9844, 0.7777, 0.7520, 0.6885, 0.6967, 0.6639]

ind = np.arange(len(f1_train))
width = 0.4

fig = plt.figure(figsize=(5, 1))

plt.bar(ind, percentage_above_th, width, color='g', alpha=0.6)
plt.xticks(ticks=ind, labels=np.arange(1, 11))
plt.yticks(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=[0, 0.2, 0.4, 0.6, 0.8, 1.0])

plt.minorticks_on()
plt.xlabel("Annotator", fontsize=10)
plt.ylabel("Percentage", fontsize=10)
plt.axis([-0.5, 9.5, 0.00, 1.0])

plt.plot(np.arange(-1, 12), np.ones(len(np.arange(-1, 12)))*0.7514, 'r', linestyle='dashed')

plt.savefig('samples_above_threshold' + '.png', dpi=200, format='png', bbox_inches='tight')

plt.show()


# 3. Uncertainty per each class

uncertainty_per_class = [0.9277, 0.7081, 0.8820, 0.6618, 0.8964, 0.6781, 0.4666]
classes = ['lm', 'lms', 'df', 'dfs', 'mfc', 'fx', 'cef']

ind = np.arange(len(uncertainty_per_class))
width = 0.4

fig = plt.figure(figsize=(5, 1))

plt.bar(ind, uncertainty_per_class, width, color='b', alpha=0.6)
plt.xticks(ticks=ind, labels=classes)
plt.yticks(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=[0, 0.2, 0.4, 0.6, 0.8, 1.0])

plt.minorticks_on()
plt.xlabel("Class", fontsize=10)
plt.ylabel("Confidence", fontsize=10)
plt.axis([-0.5, 6.5, 0.00, 1.0])

plt.plot(np.arange(-1, 12), np.ones(len(np.arange(-1, 12)))*0.7458, 'r', linestyle='dashed')

plt.savefig('uncertainty_per_class' + '.png', dpi=200, format='png', bbox_inches='tight')

plt.show()






## Histogram plot

import pandas as pd
from utils import evaluate, select_dataframe
import numpy as np
import seaborn as sn
from sklearn.metrics import plot_confusion_matrix, confusion_matrix
import matplotlib.pyplot as plt

classes = ['lm', 'lms', 'df', 'dfs', 'mfc', 'fx', 'cef']

confidence_all = []
for annotator in np.arange(1, 11):

    dir_dataframe = '../data/'
    classes = ['lm', 'lms', 'df', 'dfs', 'mfc', 'fx', 'cef']

    # Load dataframe
    data_frame = pd.read_excel(dir_dataframe + 'dataframe.xls', sheet_name='Marker_' + str(annotator))
    dataframe_test = pd.read_excel(dir_dataframe + 'training.xls')
    dataframe_test = select_dataframe(data_frame, dataframe_test)

    Y_expert = np.array(dataframe_test['GT'])
    Y_annotator = dataframe_test[classes].values/100
    confidence = np.max(Y_annotator, 1)

    confidence_all.append(confidence)

confidence_all = np.concatenate(confidence_all).flatten()

h = np.histogram(confidence_all, bins=5, range=[0, 1], density=False)
y = np.concatenate([np.zeros(1), h[0]])


plt.style.use('seaborn-white')

fig = plt.figure(figsize=(5, 3))

plt.plot(h[1], y/np.sum(y), '-', color='royalblue')

plt.minorticks_on()
plt.xlabel("Confidence", fontsize=14)
plt.ylabel("Probability", fontsize=14)
plt.axis([0., 1., 0.0, 1.0])

plt.savefig('distribution_labels' + '.png', dpi=200, format='png', bbox_inches='tight')

plt.show()