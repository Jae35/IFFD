import torch
import torch.nn as nn
import torch.nn.functional as F
from iterative_normalization import IterNormRotation as cw_layer
import pretrainedmodels


def freeze_model(model, layername):
    # Make all layers following the layername layer trainable
    ct = []
    found = False
    for name, child in model.named_children():
        if layername in ct:
            found = True
            for params in child.parameters():
                params.requires_grad = True
        ct.append(name)
    if not found:
        raise Exception('Layer not found, cant finetune!'.format(layername))
    return model

def return_pytorch04_xception():
    model = pretrainedmodels.models.xception(pretrained=False)
    # Load model in torch 0.4+
    model.fc  = model.last_linear
    del model.last_linear
    state_dict = torch.load('./models/xception-b5690688.pth')
    for name, weights in state_dict.items():
        if 'pointwise' in name:
            state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
    model.load_state_dict(state_dict)
    model.last_linear = model.fc
    del model.fc
    return model

class XceptionTransfer(nn.Module):
    def __init__(self, whitened_layers, momentum):
        super(XceptionTransfer, self).__init__()
        self.model = return_pytorch04_xception()
        self.model.last_linear = nn.Linear(self.model.last_linear.in_features, 2)

        self.whitened_layers = whitened_layers
        for whitened_layer in self.whitened_layers:
            if whitened_layer == 0:
                self.model.bn2 = cw_layer(64, momentum=momentum)
            elif whitened_layer == 1:
                self.model.block1.skipbn = cw_layer(128, momentum=momentum)
            elif whitened_layer == 2:
                self.model.block2.skipbn = cw_layer(256, momentum=momentum)
            elif whitened_layer == 3:
                self.model.block3.skipbn = cw_layer(728, momentum=momentum)
            elif whitened_layer == 4:
                self.model.block4.rep[2] = cw_layer(728, momentum=momentum)
            elif whitened_layer == 5:
                self.model.block5.rep[2] = cw_layer(728, momentum=momentum)
            elif whitened_layer == 6:
                self.model.block6.rep[2] = cw_layer(728, momentum=momentum)
            elif whitened_layer == 7:
                self.model.block7.rep[2] = cw_layer(728, momentum=momentum)
            elif whitened_layer == 8:
                self.model.block8.rep[2] = cw_layer(728, momentum=momentum)
            elif whitened_layer == 9:
                self.model.block9.rep[2] = cw_layer(728, momentum=momentum)
            elif whitened_layer == 10:
                self.model.block10.rep[2] = cw_layer(728, momentum=momentum)
            elif whitened_layer == 11:
                self.model.block11.rep[2] = cw_layer(728, momentum=momentum)
            elif whitened_layer == 12:
                self.model.block12.skipbn = cw_layer(1024, momentum=momentum)
            elif whitened_layer == 13:
                self.model.bn3 = cw_layer(1536, momentum=momentum)
            elif whitened_layer == 14:
                self.model.bn4 = cw_layer(2048, momentum=momentum)

        #self.model = freeze_model(self.model, 'conv4')

    def change_mode(self, mode):
        """
        Change the training mode
        mode = -1, no update for gradient matrix G
             = 0 to k-1, the column index of gradient matrix G that needs to be updated
        """
        for whitened_layer in self.whitened_layers:
            if whitened_layer == 1:
                self.model.block1.skipbn.mode = mode
            elif whitened_layer == 2:
                self.model.block2.skipbn.mode = mode
            elif whitened_layer == 3:
                self.model.block3.skipbn.mode = mode
            elif whitened_layer == 4:
                self.model.block4.rep[2].mode = mode
            elif whitened_layer == 5:
                self.model.block5.rep[2].mode = mode
            elif whitened_layer == 6:
                self.model.block6.rep[2].mode = mode
            elif whitened_layer == 7:
                self.model.block7.rep[2].mode = mode
            elif whitened_layer == 8:
                self.model.block8.rep[2].mode = mode
            elif whitened_layer == 9:
                self.model.block9.rep[2].mode = mode
            elif whitened_layer == 10:
                self.model.block10.rep[2].mode = mode
            elif whitened_layer == 11:
                self.model.block11.rep[2].mode = mode
            elif whitened_layer == 12:
                self.model.block12.skipbn.mode = mode
            elif whitened_layer == 13:
                self.model.bn3.mode = mode
            elif whitened_layer == 14:
                self.model.bn4.mode = mode

    def update_rotation_matrix(self):
        """
        update the rotation R using accumulated gradient G
        """
        for whitened_layer in self.whitened_layers:
            if whitened_layer == 1:
                self.model.block1.skipbn.update_rotation_matrix()
            elif whitened_layer == 2:
                self.model.block2.skipbn.update_rotation_matrix()
            elif whitened_layer == 3:
                self.model.block3.skipbn.update_rotation_matrix()
            elif whitened_layer == 4:
                self.model.block4.rep[2].update_rotation_matrix()
            elif whitened_layer == 5:
                self.model.block5.rep[2].update_rotation_matrix()
            elif whitened_layer == 6:
                self.model.block6.rep[2].update_rotation_matrix()
            elif whitened_layer == 7:
                self.model.block7.rep[2].update_rotation_matrix()
            elif whitened_layer == 8:
                self.model.block8.rep[2].update_rotation_matrix()
            elif whitened_layer == 9:
                self.model.block9.rep[2].update_rotation_matrix()
            elif whitened_layer == 10:
                self.model.block10.rep[2].update_rotation_matrix()
            elif whitened_layer == 11:
                self.model.block11.rep[2].update_rotation_matrix()
            elif whitened_layer == 12:
                self.model.block12.skipbn.update_rotation_matrix()
            elif whitened_layer == 13:
                self.model.bn3.update_rotation_matrix()
            elif whitened_layer == 14:
                self.model.bn4.update_rotation_matrix()

    def forward(self, x):
        return self.model(x)

class XceptionBN(nn.Module):
    def __init__(self):
        super(XceptionBN, self).__init__()
        self.model = return_pytorch04_xception()
        self.model.last_linear = nn.Linear(self.model.last_linear.in_features, 2)
        #self.model = freeze_model(self.model, 'conv4')

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    model = XceptionBN()
    model.load_state_dict(torch.load('./checkpoints/xception-best.pth'))
    model.eval()
    print(model.model)
