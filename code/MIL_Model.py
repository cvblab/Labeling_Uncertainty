import torch
import torch.nn as nn
import torchvision
import numpy as np
import torchvision.models as models


#####################################################
#############  MIL WITH MIL_AGGREGATOR  #############
class MILModel(torch.nn.Module):
    # def __init__(self, classes, backbone='VGG16'):
    def __init__(self, input_shape, n_classes):
        super(MILModel, self).__init__()

        # self.classes = classes
        # self.backbone = backbone
        self.input_shape = input_shape

        self.backbone_extractor = AI4SKINClassifier(in_channels=self.input_shape[0], n_classes=1)
        # self.mil_aggregator = MILAggregation()

        self.linear_layers = torch.nn.Sequential(
            torch.nn.Linear(512, n_classes)
        )
        # self.fc = torch.nn.Linear(512, 2)
        # self.classification=torch.sigmoid()
        self.aggregation = 'average'

    def forward(self, images):
        features = self.backbone_extractor(images)
        if self.aggregation == 'max':
            global_classification = torch.max(features, dim=0)[0]
        elif self.aggregation == 'average':
            global_classification = torch.mean(features, dim=0)
        elif self.aggregation == 'attention':
            global_classification = self.mil_aggregator(features)
        global_classification = self.linear_layers(global_classification)
        return global_classification



######################################################################
################## MIL WITH VISUAL TRANSFORMER #######################

class TransMIL(nn.Module):
    def __init__(self, n_classes):
        super(TransMIL, self).__init__()
        self.pos_layer = PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.n_classes)
        self.backbone_extractor = AI4SKINClassifier(in_channels=3, n_classes=1)

    def forward(self, x):
        #h = kwargs['data'].float()  # [B, n, 1024]
        h=self.backbone_extractor(x)
        #h = self._fc1(h)  # [B, n, 512]

        # ---->pad
        h = h.reshape(1, h.shape[0], -1)
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 512]

        # ---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]

        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]

        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]

        # ---->cls_token
        h = self.norm(h)[:, 0]

        # ---->predict
        logits = self._fc2(h)  # [B, n_classes]
        #Y_hat = torch.argmax(logits, dim=1)
        #Y_prob = F.softmax(logits, dim=1)
        #results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return logits

######################## Feature extraction #############################
#########################################################################
class AI4SKINClassifier(torch.nn.Module):
    def __init__(self, in_channels, n_classes=1, n_blocks=4):
        super(AI4SKINClassifier, self).__init__()
        self.n_blocks = n_blocks
        self.n_classes = n_classes
        self.output_layer = 'avgpool'

        #self.pretrained = models.resnet50(pretrained=True)
        self.pretrained = models.vgg16(pretrained=True)
        # set_parameter_requires_grad(model, True)
        print("1: ", self.pretrained)
        self.children_list = []

        ct = 0
        for n,c in self.pretrained.named_children():
            print("n: ", n)
            print("c: ", c)
            if n == 'avgpool':
                break
            elif n == 'layer1' or n=='layer2':
                # for el in c:
                for param in c.parameters():
                    param.requires_grad = False
                # self.children_list.append(c)
            else:
                # for el in c:
                for param in c.parameters():
                    param.requires_grad = True
            self.children_list.append(c)

        self.model = torch.nn.Sequential(*self.children_list,
            #torch.nn.BatchNorm2d(2048),
            #torch.nn.Conv2d(2048,1024,kernel_size=(1,1), stride=1),
            #torch.nn.ReLU(),
            #torch.nn.Conv2d(1024,512,kernel_size=(1,1), stride=1),
            #torch.nn.BatchNorm2d(512),
            torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)) )


    def forward(self, x):

        x = self.model(x)
        x = torch.squeeze(x)
        return x


class MILAggregation(torch.nn.Module):
    def __init__(self):
        super(MILAggregation, self).__init__()

        # Attention MIL
        # Attention embedding from Ilse et al. (2018) for MIL.
        # Class based on Julio Silva's MILAggregation class in PyTorch
        self.L = 512
        self.D = 128
        self.K = 1
        self.attention_V = torch.nn.Sequential(
            torch.nn.Linear(self.L, self.D),
            torch.nn.Tanh()
        )
        self.attention_U = torch.nn.Sequential(
            torch.nn.Linear(self.L, self.D),
            torch.nn.Sigmoid()
        )

        self.attention_weights = torch.nn.Linear(self.D, self.K)

    def forward(self, features):
        # Attention embedding from Ilse et al. (2018)

        # Attention weights computation
        A_V = self.attention_V(features)  # Attention
        A_U = self.attention_U(features)  # Gate
        w = torch.softmax(self.attention_weights(A_V * A_U), dim=0)  # Probabilities - softmax over instances

        # Weighted average computation per class
        # print("-- features shape:", features.shape)
        features = torch.transpose(features, 1, 0)
        # print("-- features shape B:", features.shape)
        embedding = torch.squeeze(torch.mm(features, w))  # KxL

        return embedding




class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512, dropout=0.1): #voy a utilizar una VGG asi que la dimensión es 512
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=dropout
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim) # estudiar porque se utilizan estas convoluciones
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


