import torch, time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AbsLoss(object):
    r"""An abstract class for loss functions. 
    """
    def __init__(self):
        self.record = []
        self.bs = []
    
    def compute_loss(self, pred, gt):
        r"""Calculate the loss.
        
        Args:
            pred (torch.Tensor): The prediction tensor.
            gt (torch.Tensor): The ground-truth tensor.

        Return:
            torch.Tensor: The loss.
        """
        pass
    
    def _update_loss(self, pred, gt):
        loss = self.compute_loss(pred, gt)
        self.record.append(loss.item())  # type: ignore
        self.bs.append(pred.size()[0])
        return loss
    
    def _average_loss(self):
        record = np.array(self.record)
        bs = np.array(self.bs)
        return (record*bs).sum()/bs.sum()
    
    def _reinit(self):
        self.record = []
        self.bs = []
        
class CELoss(AbsLoss):
    r"""The cross-entropy loss function.
    """
    def __init__(self):
        super(CELoss, self).__init__()
        
        self.loss_fn = nn.CrossEntropyLoss()
        
    def compute_loss(self, pred, gt):
        r"""
        """
        loss = self.loss_fn(pred, gt)
        return loss

class KLDivLoss(AbsLoss):
    r"""The Kullback-Leibler divergence loss function.
    """
    def __init__(self):
        super(KLDivLoss, self).__init__()
        
        self.loss_fn = nn.KLDivLoss()
        
    def compute_loss(self, pred, gt):
        r"""
        """
        loss = self.loss_fn(pred, gt)
        return loss

class L1Loss(AbsLoss):
    r"""The Mean Absolute Error (MAE) loss function.
    """
    def __init__(self):
        super(L1Loss, self).__init__()
        
        self.loss_fn = nn.L1Loss()
        
    def compute_loss(self, pred, gt):
        r"""
        """
        loss = self.loss_fn(pred, gt)
        return loss

class MSELoss(AbsLoss):
    r"""The Mean Squared Error (MSE) loss function.
    """
    def __init__(self):
        super(MSELoss, self).__init__()
        
        self.loss_fn = nn.MSELoss()
        
    def compute_loss(self, pred, gt):
        r"""
        """
        loss = self.loss_fn(pred, gt)
        return loss


class SoftLoss_5(AbsLoss):
    def __init__(self):
        super(SoftLoss_5, self).__init__()

        self.loss_fn = self.loss_function

    def true_metric_loss(self, true, no_of_classes, scale=1):
        batch_size = true.size(0)
        true = true.view(batch_size,1)
        true_labels = torch.cuda.LongTensor(true).repeat(1, no_of_classes).float()
        class_labels = torch.arange(no_of_classes).float().cuda()
        phi = (scale * torch.abs(class_labels - true_labels)).cuda()
        y = nn.Softmax(dim=1)(-phi)
        
        return y

    def loss_function(self, output, labels, use_soft, expt_type, device, scale):
        if use_soft:
            targets = self.true_metric_loss(labels, expt_type, scale)
            loss = torch.sum(- targets * F.log_softmax(output, -1), -1).mean()
        else:
            prob = F.softmax(output, dim=-1)
            log_prob = torch.log(prob + 1e-9)
            main_y = [torch.tensor(np.eye(5)[int(s)]) for s in labels]
            main_y = torch.stack(main_y).to(device)
            loss = torch.mean(torch.sum(-log_prob * main_y, 1))
        return loss

    def compute_loss(self, pred, gt):
        r"""
        """
        loss = self.loss_fn( output=pred, 
                                labels=gt, 
                                use_soft=True, 
                                expt_type=5, 
                                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
                                scale=2)
        return loss
class SoftLoss_4(AbsLoss):
    def __init__(self):
        super(SoftLoss_4, self).__init__()

        self.loss_fn = self.loss_function

    def true_metric_loss(self, true, no_of_classes, scale=1):
        batch_size = true.size(0)
        true = true.view(batch_size,1)
        true_labels = torch.cuda.LongTensor(true).repeat(1, no_of_classes).float()
        class_labels = torch.arange(no_of_classes).float().cuda()
        phi = (scale * torch.abs(class_labels - true_labels)).cuda()
        y = nn.Softmax(dim=1)(-phi)
        
        return y

    def loss_function(self, output, labels, use_soft, expt_type, device, scale):
        if use_soft:
            targets = self.true_metric_loss(labels, expt_type, scale)
            loss = torch.sum(- targets * F.log_softmax(output, -1), -1).mean()
        else:
            prob = F.softmax(output, dim=-1)
            log_prob = torch.log(prob + 1e-9)
            main_y = [torch.tensor(np.eye(5)[int(s)]) for s in labels]
            main_y = torch.stack(main_y).to(device)
            loss = torch.mean(torch.sum(-log_prob * main_y, 1))
        return loss

    def compute_loss(self, pred, gt):
        r"""
        """
        loss = self.loss_fn( output=pred, 
                                labels=gt, 
                                use_soft=True, 
                                expt_type=4, 
                                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
                                scale=2)
        return loss