import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers

import numpy as np
from sklearn import metrics

from LibMTL._record import _PerformanceMeter
from LibMTL.utils import count_parameters

from LibMTL.metrics import AccMetric

class Trainer(nn.Module):
    r'''A Multi-Task Learning Trainer.

    This is a unified and extensible training framework for multi-task learning. 

    Args:
        task_dict (dict): A dictionary of name-information pairs of type (:class:`str`, :class:`dict`). \
                            The sub-dictionary for each task has four entries whose keywords are named **metrics**, \
                            **metrics_fn**, **loss_fn**, **weight** and each of them corresponds to a :class:`list`.
                            The list of **metrics** has ``m`` strings, repersenting the name of ``m`` metrics \
                            for this task. The list of **metrics_fn** has two elements, i.e., the updating and score \
                            functions, meaning how to update thoes objectives in the training process and obtain the final \
                            scores, respectively. The list of **loss_fn** has ``m`` loss functions corresponding to each \
                            metric. The list of **weight** has ``m`` binary integers corresponding to each \
                            metric, where ``1`` means the higher the score is, the better the performance, \
                            ``0`` means the opposite.                           
        weighting (class): A weighting strategy class based on :class:`LibMTL.weighting.abstract_weighting.AbsWeighting`.
        architecture (class): An architecture class based on :class:`LibMTL.architecture.abstract_arch.AbsArchitecture`.
        encoder_class (class): A neural network class.
        decoders (dict): A dictionary of name-decoder pairs of type (:class:`str`, :class:`torch.nn.Module`).
        rep_grad (bool): If ``True``, the gradient of the representation for each task can be computed.
        multi_input (bool): Is ``True`` if each task has its own input data, otherwise is ``False``. 
        optim_param (dict): A dictionary of configurations for the optimizier.
        scheduler_param (dict): A dictionary of configurations for learning rate scheduler. \
                                 Set it to ``None`` if you do not use a learning rate scheduler.
        kwargs (dict): A dictionary of hyperparameters of weighting and architecture methods.

    .. note::
            It is recommended to use :func:`LibMTL.config.prepare_args` to return the dictionaries of ``optim_param``, \
            ``scheduler_param``, and ``kwargs``.

    Examples::
        
        import torch.nn as nn
        from LibMTL import Trainer
        from LibMTL.loss import CE_loss_fn
        from LibMTL.metrics import acc_update_fun, acc_score_fun
        from LibMTL.weighting import EW
        from LibMTL.architecture import HPS
        from LibMTL.model import ResNet18
        from LibMTL.config import prepare_args

        task_dict = {'A': {'metrics': ['Acc'],
                           'metrics_fn': [acc_update_fun, acc_score_fun],
                           'loss_fn': [CE_loss_fn],
                           'weight': [1]}}
        
        decoders = {'A': nn.Linear(512, 31)}
        
        # You can use command-line arguments and return configurations by ``prepare_args``.
        # kwargs, optim_param, scheduler_param = prepare_args(params)
        optim_param = {'optim': 'adam', 'lr': 1e-3, 'weight_decay': 1e-4}
        scheduler_param = {'scheduler': 'step'}
        kwargs = {'weight_args': {}, 'arch_args': {}}

        trainer = Trainer(task_dict=task_dict,
                          weighting=EW,
                          architecture=HPS,
                          encoder_class=ResNet18,
                          decoders=decoders,
                          rep_grad=False,
                          multi_input=False,
                          optim_param=optim_param,
                          scheduler_param=scheduler_param,
                          **kwargs)

    '''
    def __init__(self, task_dict, weighting, architecture, encoder_class, decoders, 
                 rep_grad, multi_input, optim_param, scheduler_param, **kwargs):
        super(Trainer, self).__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # GPU

        # 其他配置参数
        self.kwargs = kwargs                        # 其他设置参数
        self.task_dict = task_dict                  # 包含任务名和任务损失函数等配置的字典
        self.task_num = len(task_dict)              # 任务数
        self.task_name = list(task_dict.keys())     # 任务名列表
        self.rep_grad = rep_grad                    # 
        self.multi_input = multi_input              # 数据输入是否有多个数据集

        # 构建多任务学习的模型，包含权重函数、多任务学习的架构、特征提取架构、每个任务对应的分类器
        self._prepare_model(weighting, architecture, encoder_class, decoders)
        # 构建任务对应的优化器和权重衰减
        self._prepare_optimizer(optim_param, scheduler_param)
        # 计算评价指标等的一些帮助函数在里面
        self.meter = _PerformanceMeter(self.task_dict, self.multi_input)
        # 添加代码
        self.meter_val = _PerformanceMeter(self.task_dict, self.multi_input)
        
    def _prepare_model(self, weighting, architecture, encoder_class, decoders):
        """构建多任务学习模型

        Args:
            weighting (ABSWeighting): 计算多任务每个任务的权重的方式
            architecture (AbsArchitecture): _description_
            encoder_class (model): 
            decoders (model list): 多任务学习的方式。
        """
        
        class MTLmodel(architecture, weighting):
            def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device, kwargs):
                super(MTLmodel, self).__init__(task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs)
                self.init_param()
                
        self.model = MTLmodel(task_name=self.task_name, 
                              encoder_class=encoder_class, 
                              decoders=decoders, 
                              rep_grad=self.rep_grad, 
                              multi_input=self.multi_input,
                              device=self.device,
                              kwargs=self.kwargs['arch_args']).to(self.device)
        count_parameters(self.model)
        
    def _prepare_optimizer(self, optim_param, scheduler_param):
        optim_dict = {
                'sgd': torch.optim.SGD,
                'adam': torch.optim.Adam,
                'adagrad': torch.optim.Adagrad,
                'rmsprop': torch.optim.RMSprop,
                'adamw': torch.optim.AdamW,
            }
        scheduler_dict = {
                'exp': torch.optim.lr_scheduler.ExponentialLR,
                'step': torch.optim.lr_scheduler.StepLR,
                'cos': torch.optim.lr_scheduler.CosineAnnealingLR,
            }
        optim_arg = {k: v for k, v in optim_param.items() if k != 'optim'}
        self.optimizer = optim_dict[optim_param['optim']](self.model.parameters(), **optim_arg)
        if scheduler_param is not None:
            scheduler_arg = {k: v for k, v in scheduler_param.items() if k != 'scheduler'}
            self.scheduler = scheduler_dict[scheduler_param['scheduler']](self.optimizer, **scheduler_arg)
        else:
            self.scheduler = None

    def _process_data(self, loader):
        # try:
        #     data, label = loader[1].next()
        # except:
        #     loader[1] = iter(loader[0])
        #     data, label = loader[1].next()
        # data = data.to(self.device)
        # if not self.multi_input:
        #     for task in self.task_name:
        #         label[task] = label[task].to(self.device)
        # else:
        #     label = label.to(self.device)
        # return data, label

        try:
            data, label = loader[1].next()
        except:
            loader[1] = iter(loader[0])
            data, label = loader[1].next()
        for i in range(len(data)):    
            data[i] = data[i].to(self.device)
        if not self.multi_input:
            for task in self.task_name:
                label[task] = label[task].to(self.device)
        else:
            label = label.to(self.device)
        return data, label
    
    def process_preds(self, preds, task_name=None):
        r'''The processing of prediction for each task. 

        - The default is no processing. If necessary, you can rewrite this function. 
        - If ``multi_input`` is ``True``, ``task_name`` is valid and ``preds`` with type :class:`torch.Tensor` is the prediction of this task.
        - otherwise, ``task_name`` is invalid and ``preds`` is a :class:`dict` of name-prediction pairs of all tasks.

        Args:
            preds (dict or torch.Tensor): The prediction of ``task_name`` or all tasks.
            task_name (str): The string of task name.
        '''
        return preds

    def _compute_loss(self, preds, gts, task_name=None):
        if not self.multi_input:
            train_losses = torch.zeros(self.task_num).to(self.device)
            for tn, task in enumerate(self.task_name):
                # self.meter.losses[task]._update_loss:在定义任务时，就已经制定了loss函数，
                # loss函数的抽象类有_update_loss()接口
                train_losses[tn] = self.meter.losses[task]._update_loss(preds[task], gts[task])
        else:
            train_losses = self.meter.losses[task_name]._update_loss(preds, gts)
        return train_losses
        
    def _prepare_dataloaders(self, dataloaders):
        if not self.multi_input:
            loader = [dataloaders, iter(dataloaders)]
            return loader, len(dataloaders)
        else:
            loader = {}
            batch_num = []
            for task in self.task_name:
                loader[task] = [dataloaders[task], iter(dataloaders[task])]
                batch_num.append(len(dataloaders[task]))
            return loader, batch_num

    def train(self, train_dataloaders, test_dataloaders, epochs, 
              val_dataloaders=None, return_weight=False):
        r'''The training process of multi-task learning.

        Args:
            train_dataloaders (dict or torch.utils.data.DataLoader): The dataloaders used for training. \
                            If ``multi_input`` is ``True``, it is a dictionary of name-dataloader pairs. \
                            Otherwise, it is a single dataloader which returns data and a dictionary \
                            of name-label pairs in each iteration.

            test_dataloaders (dict or torch.utils.data.DataLoader): The dataloaders used for the validation or testing. \
                            The same structure with ``train_dataloaders``.
            epochs (int): The total training epochs.
            return_weight (bool): if ``True``, the loss weights will be returned.
        '''
        # 添加代码，为了记录每个batch的loss和Acc
        acc = AccMetric()
        count = 0


        train_loader, train_batch = self._prepare_dataloaders(train_dataloaders)
        train_batch = min(train_batch) if self.multi_input else train_batch  # type: ignore
        # 修改代码，数据集batch相差太大，试下用最小值。
        # train_batch = max(train_batch) if self.multi_input else train_batch  # type: ignore
        
        self.batch_weight = np.zeros([self.task_num, epochs, train_batch])  # type: ignore
        self.model.train_loss_buffer = np.zeros([self.task_num, epochs])
        self.model.train_acc_buffer = np.zeros([self.task_num, epochs])
        self.model.epochs = epochs
        # 添加代码，为了存储每一个batch的loss
        loss_all = np.array([], dtype=float)

        for epoch in range(epochs):
            self.model.epoch = epoch
            self.model.train()
            self.meter.record_time('begin')
            # ----修改过----直接添加
            # train_loader, _ = self._prepare_dataloaders(train_dataloaders)
            # -------------
            for batch_index in range(train_batch):  # type: ignore
                if not self.multi_input:
                    train_inputs, train_gts = self._process_data(train_loader)
                    train_preds = self.model(train_inputs)
                    train_preds = self.process_preds(train_preds)
                    train_losses = self._compute_loss(train_preds, train_gts)
                    self.meter.update(train_preds, train_gts)
                else:
                    train_losses = torch.zeros(self.task_num).to(self.device)
                    for tn, task in enumerate(self.task_name):
                        train_input, train_gt = self._process_data(train_loader[task])
                        train_pred = self.model(train_input, task)
                        train_pred = train_pred[task]
                        train_pred = self.process_preds(train_pred, task)
                        train_losses[tn] = self._compute_loss(train_pred, train_gt, task)
                        self.meter.update(train_pred, train_gt, task)

                        # 添加代码
                        # acc.update_fun(train_pred, train_gt)
                        # if count % 1000 == 0:
                        #     self.val(test_dataloaders, epoch, mode='test')
                            # print('\t{}: loss:{}\tAcc:{}'.format(task, train_losses[tn], acc.score_fun()), end='')
                            # acc.reinit()
                    # count += 1
                    # if count % 100 == 0:
                    #     print()
                # 添加代码
                # loss_all = np.append(loss_all, train_losses.cpu().detach().numpy())

                self.optimizer.zero_grad()
                w = self.model.backward(train_losses, **self.kwargs['weight_args'])
                if w is not None:
                    self.batch_weight[:, epoch, batch_index] = w
                self.optimizer.step()
            
            self.meter.record_time('end')
            self.meter.get_score()
            self.model.train_loss_buffer[:, epoch] = self.meter.loss_item
            # 添加代码，增加存储acc的代码
            self.model.train_acc_buffer[:, epoch] = self.meter.acc_item

            self.meter.display(epoch=epoch, mode='train')
            self.meter.reinit()
            
            if val_dataloaders is not None:
                self.meter.has_val = True
                self.test(val_dataloaders, epoch, mode='val')
            self.test(test_dataloaders, epoch, mode='test')
            # 添加代码
            # self.test_all_label(test_dataloaders=test_all_6)

            if self.scheduler is not None:
                self.scheduler.step()
        # self.meter.display_best_result()
        best_result = self.meter.display_best_result()
        # if return_weight:
        return self.batch_weight, best_result
        
        # # 添加代码
        # # 要修改路径
        # import pickle
        # if not os.path.exists('./loss'):
        #     os.makedirs('./loss')
        # with open('./loss/selfharm_loss.pkl', 'wb') as f:
        #     pickle.dump(self.model.train_loss_buffer, f)
        # with open('./loss/selfharm_acc.pkl', 'wb') as f:
        #     pickle.dump(self.model.train_acc_buffer, f)


    def test(self, test_dataloaders, epoch=None, mode='test'):
        r'''The test process of multi-task learning.

        Args:
            test_dataloaders (dict or torch.utils.data.DataLoader): If ``multi_input`` is ``True``, \
                            it is a dictionary of name-dataloader pairs. Otherwise, it is a single \
                            dataloader which returns data and a dictionary of name-label pairs in each iteration.
            epoch (int, default=None): The current epoch. 
        '''
        
        test_loader, test_batch = self._prepare_dataloaders(test_dataloaders)

        self.model.eval()
        self.meter.record_time('begin')
        with torch.no_grad():
            if not self.multi_input:
                for batch_index in range(test_batch):  # type: ignore
                    test_inputs, test_gts = self._process_data(test_loader)
                    test_preds = self.model(test_inputs)
                    test_preds = self.process_preds(test_preds)
                    test_losses = self._compute_loss(test_preds, test_gts)
                    self.meter.update(test_preds, test_gts)
            else:
                for tn, task in enumerate(self.task_name):
                    for batch_index in range(test_batch[tn]):  # type: ignore
                        test_input, test_gt = self._process_data(test_loader[task])
                        test_pred = self.model(test_input, task)
                        test_pred = test_pred[task]
                        test_pred = self.process_preds(test_pred)
                        test_loss = self._compute_loss(test_pred, test_gt, task)
                        self.meter.update(test_pred, test_gt, task)
        self.meter.record_time('end')
        self.meter.get_score()
        self.meter.display(epoch=epoch, mode=mode)
        self.meter.reinit()

    def val(self, test_dataloaders, epoch=None, mode='val'):
        r'''The test process of multi-task learning.

        Args:
            test_dataloaders (dict or torch.utils.data.DataLoader): If ``multi_input`` is ``True``, \
                            it is a dictionary of name-dataloader pairs. Otherwise, it is a single \
                            dataloader which returns data and a dictionary of name-label pairs in each iteration.
            epoch (int, default=None): The current epoch. 
        '''

        test_loader, test_batch = self._prepare_dataloaders(test_dataloaders)
        self.model.eval()
        self.meter_val.record_time('begin')
        with torch.no_grad():
            if not self.multi_input:
                for batch_index in range(test_batch):  # type: ignore
                    test_inputs, test_gts = self._process_data(test_loader)
                    test_preds = self.model(test_inputs)
                    test_preds = self.process_preds(test_preds)
                    test_losses = self._compute_loss(test_preds, test_gts)
                    self.meter_val.update(test_preds, test_gts)
            else:
                for tn, task in enumerate(self.task_name):
                    for batch_index in range(test_batch[tn]):  # type: ignore
                        test_input, test_gt = self._process_data(test_loader[task])
                        test_pred = self.model(test_input, task)
                        test_pred = test_pred[task]
                        test_pred = self.process_preds(test_pred)
                        test_loss = self._compute_loss(test_pred, test_gt, task)
                        self.meter_val.update(test_pred, test_gt, task)
        self.meter_val.record_time('end')
        self.meter_val.get_score()
        self.meter_val.display(epoch=epoch, mode=mode, is_val=True)
        self.meter_val.reinit()

    def process_preds_all_label(self, preds, task_name):

        label_dir = {'0':5, '1': 0, '2': 5, '3': 1, '4': 5, '5': 2, '6': 5, '7': 3,
                     '8': 5, '9': 4}
        for i in preds.keys():
            preds[i] = preds[i].detach().cpu().numpy()
        true_label = np.array([], dtype=int)
        for i in range(list(preds.values())[0].__len__()):
            tag_res = np.array([], dtype=float)

            for tn, task in enumerate(task_name):
                tag_res = np.append( tag_res, preds[task][i])
            
            true_label = np.append(true_label, label_dir[str(np.argmax(tag_res))])

        return true_label

    def _prepare_dataloaders_6(self, dataloaders):
        loader = list(dataloaders)
        return loader, len(dataloaders[0])
        


    def test_all_label(self, test_dataloaders):
        r'''The test process of multi-task learning.

        Args:
            test_dataloaders (dict or torch.utils.data.DataLoader): If ``multi_input`` is ``True``, \
                            it is a dictionary of name-dataloader pairs. Otherwise, it is a single \
                            dataloader which returns data and a dictionary of name-label pairs in each iteration.
            epoch (int, default=None): The current epoch. 
        '''
        test_loader, test_batch = self._prepare_dataloaders_6(test_dataloaders)
        
        label_all_6 = np.array([], dtype=int)
        pred_all_6 = np.array([], dtype=int)

        self.model.eval()
        self.meter.record_time('begin')
        with torch.no_grad():
            for batch_index in range(test_batch):  # type: ignore
                test_inputs, test_gts = self._process_data(test_loader)
                test_preds = self.model(test_inputs)
                test_preds = self.process_preds_all_label(test_preds, self.task_name)
                label_all_6 = np.append(label_all_6, test_gts.cpu().numpy())
                pred_all_6 = np.append(pred_all_6, test_preds)
                # test_losses = nn.CrossEntropyLoss(test_preds, test_gts)
                # self.meter.update(test_preds, test_gts)
        acc = metrics.accuracy_score(label_all_6, pred_all_6)
        target_name = ['adhd', 'anxiety', 'bipolar', 'depression', 'ptsd', 'none']
        report = metrics.classification_report(label_all_6, pred_all_6, target_names=target_name, digits=4)
        confusion = metrics.confusion_matrix(label_all_6, pred_all_6)
        evaluation_index = []
        evaluation_index.append(str('\n' + str(acc) + '\n'))
        evaluation_index.append(report)
        evaluation_index.append(str(confusion))

        # 经常用到的(如果文件夹不存在，则创建该文件夹)
        if not os.path.exists('./result'):
            os.makedirs('./result')
        with open('./reslut/self_harm.txt', 'a+') as f:
            f.writelines(evaluation_index)



