import torch
import torch.utils.data
import torch.nn as nn

from nerfstudio.field_components.encodings import HashEncoding

def get_model(params):
    return ResidualFCNet(params['input_dim'], params['num_classes'], params['num_filts'], params['ratio'], params['depth'])

class ResLayer(nn.Module):
    def __init__(self, linear_size):
        super(ResLayer, self).__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout()
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.dropout1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out = x + y
        return out

class ResidualFCNet(nn.Module):

    def __init__(self, num_inputs, num_classes, num_filts, mlp_ratio, depth=4):
        super(ResidualFCNet, self).__init__()
        num_inputs = 4
        self.inc_bias = False
        self.class_emb = nn.Linear(num_filts, num_classes, bias=self.inc_bias)
        self.num_mlp = int(num_filts * mlp_ratio)
        self.num_hash = num_filts - self.num_mlp
        layers = []
        layers.append(nn.Linear(num_inputs, self.num_mlp))
        layers.append(nn.ReLU(inplace=True))
        for i in range(depth):
            layers.append(ResLayer(self.num_mlp))
        self.feats = torch.nn.Sequential(*layers)

        if self.num_hash:
            chart = {32: (4, 8), 64: (8, 8), 96: (12, 8), 128: (16, 8), 160: (10, 16), 192: (12, 16), 224: (14, 16), 256: (16, 16)}
            levels, feats = chart[self.num_hash]
            try: self.hash_enc = HashEncoding(num_levels=levels, features_per_level=feats)
            except Exception: self.hash_enc = HashEncoding(num_levels=levels, features_per_level=feats, implementation = 'torch')

    def forward(self, x, class_of_interest=None, return_feats=False):
        if self.num_mlp: loc_emb1 = self.feats(x[:, :4])
        if self.num_hash: loc_emb2 = self.hash_enc(x[:, 4:])
        loc_emb = torch.cat((loc_emb1, loc_emb2), dim=1) if self.num_mlp and self.num_hash else loc_emb1 if self.num_mlp else loc_emb2
        loc_emb = loc_emb.float()
        if return_feats:
            return loc_emb
        if class_of_interest is None:
            class_pred = self.class_emb(loc_emb)
        else:
            class_pred = self.eval_single_class(loc_emb, class_of_interest)
        return torch.sigmoid(class_pred)

    def eval_single_class(self, x, class_of_interest):
        if self.inc_bias:
            return x @ self.class_emb.weight[class_of_interest, :] + self.class_emb.bias[class_of_interest]
        else:
            return x @ self.class_emb.weight[class_of_interest, :]
