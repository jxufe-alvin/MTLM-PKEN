import torch, time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from sklearn import metrics

class AbsMetric(object):
    r"""An abstract class for the performance metrics of a task. 

    Attributes:
        record (list): A list of the metric scores in every iteration.
        bs (list): A list of the number of data in every iteration.
    """
    def __init__(self):
        self.record = []
        self.bs = []
        self.label_all = np.array([], dtype=int)
        self.pred_all = np.array([], dtype=int)
    
    # @property
    def update_fun(self, pred, gt):
        r"""Calculate the metric scores in every iteration and update :attr:`record`.

        Args:
            pred (torch.Tensor): The prediction tensor.
            gt (torch.Tensor): The ground-truth tensor.
        """
        pass
    
    @property
    def score_fun(self):
        r"""Calculate the final score (when an epoch ends).

        Return:
            list: A list of metric scores.
        """
        pass
    
    def reinit(self):
        r"""Reset :attr:`record` and :attr:`bs` (when an epoch ends).
        """
        self.record = []
        self.bs = []
        """
        这里加了代码
        时间：2022.11.14
        """
        self.label_all = np.array([], dtype=int)
        self.pred_all = np.array([], dtype=int)

    
# accuracy
class AccMetric(AbsMetric):
    r"""Calculate the accuracy.
    """
    def __init__(self):
        super(AccMetric, self).__init__()
        
    def update_fun(self, pred, gt):
        r"""
        """
        pred = F.softmax(pred, dim=-1).max(-1)[1]
        self.record.append(gt.eq(pred).sum().item())
        self.bs.append(pred.size()[0])
        
    def score_fun(self):
        r"""
        """
        return [(sum(self.record)/sum(self.bs))]



class AllMetric(AbsMetric):
    def __init__(self):
        super(AllMetric, self).__init__()

    def update_fun(self, pred, gt):
        pred = F.softmax(pred, dim=-1).max(-1)[1]
        self.record.append(gt.eq(pred).sum().item())
        self.bs.append(pred.size()[0])
        self.label_all = np.append(self.label_all, gt.cpu().numpy())
        self.pred_all = np.append(self.pred_all, pred.cpu().numpy())

    def score_fun(self, task):
        acc = metrics.accuracy_score(self.label_all, self.pred_all)
        # target_name = ['0', '1', '2', '3', '4', '5']
        # target_name = ['0', '1']
        if 'adhd' in task:
            target_name = ['none', 'ADHD']
        elif 'anxiety' in task:
            target_name = ['none', 'anxiety']
        elif 'bipolar' in task:
            target_name = ['none', 'bipolar']
        elif 'depression' in task:
            target_name = ['none', 'depression']
        elif 'ptsd' in task:
            target_name = ['none', 'ptsd']
        elif task == 'mmh':
            target_name = ['adhd', 'anxiety', 'bipolar', 'depression', 'ptsd', 'none']
        elif 'self_harm' in task :
            target_name = ['green', 'amber', 'red', 'crisis']
        
        else:
            raise ValueError ('meiyou {}这个任务！'.format(task))
    
        report = metrics.classification_report(self.label_all, self.pred_all, target_names=target_name, digits=4)
        confusion = metrics.confusion_matrix(self.label_all, self.pred_all)
        return [(sum(self.record)/sum(self.bs)), acc, report, confusion]
        



# L1 Error
class L1Metric(AbsMetric):
    r"""Calculate the Mean Absolute Error (MAE).
    """
    def __init__(self):
        super(L1Metric, self).__init__()
        
    def update_fun(self, pred, gt):
        r"""
        """
        abs_err = torch.abs(pred - gt)
        self.record.append(abs_err.item())
        self.bs.append(pred.size()[0])
        
    def score_fun(self):
        r"""
        """
        records = np.array(self.record)
        batch_size = np.array(self.bs)
        return [(records*batch_size).sum()/(sum(batch_size))]
