import torch


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1, class_weight=None, train_gpu=True,
                 type_labels='hard'):
        """if smoothing == 0, it's one-hot method
           if 0 < smoothing < 1, it's smooth method
        """
        super(CrossEntropyLoss, self).__init__()
        self.smoothing = smoothing  # Recommended value: 1/K, with K the number of classes
        self.confidence = 1.0 - smoothing
        self.class_weight = class_weight
        self.cls = classes
        self.dim = dim
        self.train_gpu = train_gpu
        self.type_labels = type_labels

    def forward(self, logits, target, preact=True):
        assert 0 <= self.smoothing < 1

        if len(logits.shape) == 1:
            logits = logits.unsqueeze(0)

        # Log-Softmax activation
        if preact:
            pred = logits.log_softmax(dim=self.dim)
        else:
            pred = torch.log(logits)

        # Target from one-hot encoding to categorical
        #if target.shape[-1] > 1 and self.type_labels == 'hard':
        #    target = torch.argmax(target, dim=self.dim)

        # Apply class weights
        if self.class_weight is None:
            weight = torch.ones(self.cls).unsqueeze(0)
        else:
            classes_categorical = torch.argmax(target, dim=self.dim)
            weight = torch.tensor(self.class_weight)[classes_categorical]

        if self.train_gpu:
            weight = weight.cuda()

        pred = pred * weight

        # Label smoothing
        if self.type_labels == 'hard':
            classes_categorical = torch.argmax(target, dim=self.dim).long()

            while len(list(classes_categorical.shape)) < 2:
                classes_categorical = classes_categorical.unsqueeze(0)

            with torch.no_grad():
                true_dist = torch.zeros_like(pred)
                true_dist.fill_(self.smoothing / (self.cls - 1))
                true_dist.scatter_(1, classes_categorical, self.confidence)
        else:
            true_dist = target

        # Compute loss
        ce = torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

        return ce


class ShannonEntropy(torch.nn.Module):
    def __init__(self, dim=-1):
        super(ShannonEntropy, self).__init__()
        self.dim = dim

    def forward(self, logits):
        # Log-Softmax activation
        log_pred = logits.log_softmax(dim=self.dim)
        # Probabilities
        pred = logits.softmax(dim=self.dim)
        # Compute loss
        h = torch.mean(torch.sum(log_pred * pred, dim=self.dim))
        return h

