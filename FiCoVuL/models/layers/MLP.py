from torch import nn


class MLP3Layer(nn.Module):
    zoom_out_proportion = 2 / 3

    def __init__(self, num_in_features: int, num_out_features: int, output_activate: str = None, bias=True,
                 dropout_prob=0.2):
        super().__init__()

        hidden_dims = [num_in_features,
                       round((max(num_in_features, num_out_features) + num_out_features) * self.zoom_out_proportion),
                       num_out_features]

        self.mlp = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dims[0], hidden_dims[1], bias=bias),
            # nn.LeakyReLU(0.2),
            nn.Mish(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dims[1], hidden_dims[2], bias=bias),
        )
        if output_activate is not None:
            # self.mlp.add_module(str(len(self.mlp)), nn.Sigmoid() if output_activate.lower() == 'sigmoid' else nn.ReLU())
            self.mlp.add_module(str(len(self.mlp)), nn.Sigmoid() if output_activate.lower() == 'sigmoid' else nn.Mish())

        self.init_params()

    def init_params(self):
        for name, param in self.mlp.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            if 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, data):
        return self.mlp(data)


class MLP4Layer(nn.Module):
    zoom_out_proportion = 1.25

    def __init__(self, num_in_features: int, num_out_features: int, output_activate: str = None, bias=True,
                 dropout_prob=0.2):
        super().__init__()

        hidden_dims = [num_in_features,
                       round(max(num_in_features, num_out_features) * self.zoom_out_proportion),
                       round((max(num_in_features, num_out_features) * self.zoom_out_proportion + num_out_features) * 2 / 3),
                       num_out_features]

        self.mlp = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dims[0], hidden_dims[1], bias=bias),
            # nn.LeakyReLU(0.2),
            nn.Mish(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dims[1], hidden_dims[2], bias=bias),
            # nn.LeakyReLU(0.2),
            nn.Mish(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dims[2], hidden_dims[3], bias=bias),
        )
        if output_activate is not None:
            # self.mlp.add_module(str(len(self.mlp)), nn.Sigmoid() if output_activate.lower() == 'sigmoid' else nn.ReLU())
            self.mlp.add_module(str(len(self.mlp)), nn.Sigmoid() if output_activate.lower() == 'sigmoid' else nn.Mish())

        self.init_params()

    def init_params(self):
        for name, param in self.mlp.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            if 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, data):
        return self.mlp(data)


class MLP5Layer(nn.Module):
    zoom_out_proportion = 1.25

    def __init__(self, num_in_features: int, num_out_features: int, output_activate: str = None, bias=True,
                 dropout_prob=0.2):
        super().__init__()

        hidden_dims = [num_in_features,
                       round(max(num_in_features, num_out_features) * self.zoom_out_proportion),
                       round((max(num_in_features,
                                  num_out_features) * self.zoom_out_proportion + num_out_features) * 2 / 3),
                       round((max(num_in_features,
                                  num_out_features) * self.zoom_out_proportion + num_out_features) * 2 / 3),
                       num_out_features]

        self.mlp = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dims[0], hidden_dims[1], bias=bias),
            # nn.LeakyReLU(0.2),
            nn.Mish(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dims[1], hidden_dims[2], bias=bias),
            # nn.LeakyReLU(0.2),
            nn.Mish(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dims[2], hidden_dims[3], bias=bias),
            # nn.LeakyReLU(0.2),
            nn.Mish(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dims[3], hidden_dims[4], bias=bias),
        )
        if output_activate is not None:
            # self.mlp.add_module(str(len(self.mlp)), nn.Sigmoid() if output_activate.lower() == 'sigmoid' else nn.ReLU())
            self.mlp.add_module(str(len(self.mlp)), nn.Sigmoid() if output_activate.lower() == 'sigmoid' else nn.Mish())

        self.init_params()

    def init_params(self):
        for name, param in self.mlp.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            if 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, data):
        return self.mlp(data)


class MLP(nn.Module):
    def __init__(self, num_features, output_activate: str = None, bias=True, dropout_prob=0.2):
        super(MLP, self).__init__()
        assert len(num_features) >= 2

        temp = []
        i = 0
        for i in range(len(num_features)-2):
            temp.append(nn.Dropout(dropout_prob))
            temp.append(nn.Linear(num_features[i], num_features[i+1], bias=bias))
            # temp.append(nn.LeakyReLU(0.2))
            temp.append(nn.Mish)
        temp.append(nn.Dropout(dropout_prob))
        temp.append(nn.Linear(num_features[i], num_features[i + 1], bias=bias))

        self.mlp = nn.Sequential(*temp)
        if output_activate is not None:
            # self.mlp.add_module(str(len(self.mlp)), nn.Sigmoid() if output_activate.lower() == 'sigmoid' else nn.ReLU())
            self.mlp.add_module(str(len(self.mlp)), nn.Sigmoid() if output_activate.lower() == 'sigmoid' else nn.Mish())

        self.init_params()

    def init_params(self):
        for name, param in self.mlp.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            if 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, data):
        return self.mlp(data)
