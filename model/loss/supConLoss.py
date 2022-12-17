
import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    def __init__(self, tau=0.07):
        super(SupConLoss, self).__init__()
        self.tau = tau

    def forward(self, feature, y):
        device = torch.device('cuda') if feature.is_cuda else torch.device('cpu')

        N = y.shape[0]

        y_tilda = torch.stack((y, y))
        y_tilda = y_tilda.T.reshape(-1)

        mask_eq = torch.eq(y_tilda, y_tilda.unsqueeze(1)).int()
        mask_sign = mask_eq - torch.eye(2*N).to(device)

        # (2N) vector of i
        Ny_tilda = sum(mask_eq)

        inner_prod_term = torch.matmul(feature, feature.T)
        inner_prod_term = torch.exp(inner_prod_term / self.tau)

        # (2N) vector of i
        mom = inner_prod_term.clone()
        mom[range(len(mom)), range(len(mom))] = 0
        mom = sum(mom)

        mask_sign = mask_sign * torch.log((inner_prod_term / mom).T)
        mask_sign = -(mask_sign.T / (2*Ny_tilda - 1))

        return sum(mask_sign).mean()
