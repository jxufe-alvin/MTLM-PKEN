import torch, time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle

import utils_MTLM_PKEN as utils

from LibMTL.utils import count_improvement

# utils.init_()

class _PerformanceMeter(object):
    def __init__(self, task_dict, multi_input, base_result=None):
        
        self.task_dict = task_dict
        self.multi_input = multi_input
        self.task_num = len(self.task_dict)
        self.task_name = list(self.task_dict.keys())
        
        self.weight = {task: self.task_dict[task]['weight'] for task in self.task_name}
        self.base_result = base_result
        self.best_result = {'improvement': -1e+2, 'epoch': 0, 'result': 0}
        # 损失
        self.losses = {task: self.task_dict[task]['loss_fn'] for task in self.task_name}
        # 评价指标
        self.metrics = {task: self.task_dict[task]['metrics_fn'] for task in self.task_name}
        
        self.results = {task:[] for task in self.task_name}
        self.loss_item = np.zeros(self.task_num)

        # 添加代码
        self.acc_item = np.zeros(self.task_num)
        
        self.has_val = False

        
        self._init_display()
        
    def record_time(self, mode='begin'):
        if mode == 'begin':
            self.beg_time = time.time()
        elif mode == 'end':
            self.end_time = time.time()
        else:
            raise ValueError('No support time mode {}'.format(mode))
        
    def update(self, preds, gts, task_name=None):
        with torch.no_grad():
            if task_name is None:
                for tn, task in enumerate(self.task_name):
                    self.metrics[task].update_fun(preds[task], gts[task])
            else:
                self.metrics[task_name].update_fun(preds, gts)

    def update_batch(self, preds, gts, task_name=None):
        with torch.no_grad():
            if task_name is None:
                for tn, task in enumerate(self.task_name):
                    self.metrics[task].update_fun(preds[task], gts[task])
            else:
                self.metrics[task_name].update_fun(preds, gts)
        
    def get_score(self):
        with torch.no_grad():
            for tn, task in enumerate(self.task_name):
                # 修改过
                # self.results[task] = self.metrics[task].score_fun()
                self.results[task] = self.metrics[task].score_fun(task)
                self.loss_item[tn] = self.losses[task]._average_loss()
                self.acc_item[tn] = self.results[task][0]

    def get_score_batch(self):
        with torch.no_grad():
            for tn, task in enumerate(self.task_name):
                # 修改过
                # self.results[task] = self.metrics[task].score_fun()
                self.results[task] = self.metrics[task].score_fun(task)
                self.loss_item[tn] = self.losses[task]._average_loss()
                self.acc_item[tn] = self.results[task][0]
    
    def _init_display(self):
        print('='*40)
        print('LOG FORMAT | ', end='')
        for tn, task in enumerate(self.task_name):
            print(task+'_LOSS ', end='')
            for m in self.task_dict[task]['metrics']:
                print(m+' ', end='')
            print('| ', end='')
        print('TIME')
    
    def display(self, mode, epoch, is_val = False):
        # 修改代码，加入是否是val的参数
        if epoch == 0 and self.base_result is None and mode==('val' if self.has_val else 'test'):
            self.base_result = self.results
        if mode == 'train':
            print('Epoch: {:04d} | '.format(epoch), end='')
        if not self.has_val and mode == 'test':
            self._update_best_result(self.results, epoch)
        if self.has_val and mode != 'train':
            self._update_best_result_by_val(self.results, epoch, mode)
        if mode == 'train':
            p_mode = 'TRAIN'
        elif mode == 'val':
            p_mode = 'VAL'
        else:
            p_mode = 'TEST'
        print('{}: '.format(p_mode), end='')
        for tn, task in enumerate(self.task_name):
            print('{:.4f} '.format(self.loss_item[tn]), end='')

            # for i in range(len(self.results[task])):
            #     print('{:.4f} '.format(self.results[task][i]), end='')
            
            print('{:.4f} '.format(self.results[task][0]), end='')
            print('| ', end='')

        print('Time: {:.4f}'.format(self.end_time-self.beg_time), end='')
        print(' | ', end='') if mode!='test' else print()
        
        cur_time, params_model = utils.get_global()

        utils.record_loss_and_acc(epoch, 
                                  mode, 
                                  self.results, 
                                  self.task_name, 
                                  self.loss_item, 
                                  self.end_time, 
                                  self.beg_time, 
                                  cur_time, 
                                  params_model)


        # # 修改代码，加入是否是val的参数
        # cur_time = time.strftime('%Y-%m-%d-%H',time.localtime())
        # f = open('./loss/{}_res_selfharm_depression.txt'.format(cur_time), 'a+')
        # if epoch == 0 and self.base_result is None and mode==('val' if self.has_val else 'test'):
        #     self.base_result = self.results
        # if mode == 'train':
        #     print('Epoch: {:04d} | '.format(epoch), end='')
        #     f.write('Epoch: {:04d} | '.format(epoch))
        # if not self.has_val and mode == 'test':
        #     self._update_best_result(self.results, epoch)
        # if self.has_val and mode != 'train':
        #     self._update_best_result_by_val(self.results, epoch, mode)
        # if mode == 'train':
        #     p_mode = 'TRAIN'
        # elif mode == 'val':
        #     p_mode = 'VAL'
        # else:
        #     p_mode = 'TEST'
        # print('{}: '.format(p_mode), end='')
        # for tn, task in enumerate(self.task_name):
        #     print('{:.4f} '.format(self.loss_item[tn]), end='')
        #     f.write('{:.4f} '.format(self.loss_item[tn]))
        #     # for i in range(len(self.results[task])):
        #     #     print('{:.4f} '.format(self.results[task][i]), end='')
            
        #     print('{:.4f} '.format(self.results[task][0]), end='')
        #     f.write('{:.4f} '.format(self.results[task][0]))
        #     print('| ', end='')
        #     f.write('| ')

        # print('Time: {:.4f}'.format(self.end_time-self.beg_time), end='')
        # f.write('Time: {:.4f}'.format(self.end_time-self.beg_time))
        # print(' | ', end='') if mode!='test' else print()
        # f.write(' | ') if mode!='test' else f.write('\n')
        # f.close()

        # if mode == 'test':
        #     for tn, task in enumerate(self.task_name):
        #         print(self.results[task][1])
        #         print("Precision, Recall and F1-Score...")
        #         print(self.results[task][2])
        #         print("Confusion Matrix...")
        #         print(self.results[task][3])
        # 修改代码，加入是否是val的参数
        # 要修改路径

        if mode == 'test' and not is_val:
            utils.record_F1(epoch=epoch, 
                            results=self.results, 
                            task_name=self.task_name, 
                            cur_time=cur_time, 
                            params_model=params_model)

            # import os
            # # 经常用到的(如果文件夹不存在，则创建该文件夹)
            # if not os.path.exists('./result'):
            #     os.makedirs('./result')

            # with open(f'./result/{cur_time}_self_harm_test_{epoch}.pkl', 'wb') as f:
            #     pickle.dump(self.results, f)
            
        elif mode == 'test' and is_val:
            pass
        else:
            pass



        # if mode == 'test':
        #     with open('examples/MultiMental/result/result.csv', 'a') as f:
        #         for tn, task in enumerate(self.task_name):
        #             f.writelines([f'{task}Acc\n'])
        #             f.writelines(self.results[task][1])
        #             f.write('Precision, Recall and F1-Score...\n')
        #             f.writelines(self.results[task][2])
        #             f.write("Confusion Matrix...\n")
        #             f.writelines(self.results[task][3])

        
    def display_best_result(self):
        print('='*40)
        print('Best Result: Epoch {}, result {}'.format(self.best_result['epoch'], self.best_result['result']))
        print('='*40)
        
        # 添加代码
        return self.best_result

        
    def _update_best_result_by_val(self, new_result, epoch, mode):
        if mode == 'val':
            improvement = count_improvement(self.base_result, new_result, self.weight)
            if improvement > self.best_result['improvement']:
                self.best_result['improvement'] = improvement
                self.best_result['epoch'] = epoch
        else:
            if epoch == self.best_result['epoch']:
                self.best_result['result'] = new_result
        
    def _update_best_result(self, new_result, epoch):
        improvement = count_improvement(self.base_result, new_result, self.weight)
        if improvement > self.best_result['improvement']:
            self.best_result['improvement'] = improvement
            self.best_result['epoch'] = epoch
            self.best_result['result'] = new_result
        
    def reinit(self):
        for task in self.task_name:
            self.losses[task]._reinit()
            self.metrics[task].reinit()
        self.loss_item = np.zeros(self.task_num)
        self.acc_item = np.zeros(self.task_num)
        self.results = {task:[] for task in self.task_name}

