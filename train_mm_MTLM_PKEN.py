# import sys
# sys.path.append('./LibMTL-main')

import warnings
warnings.filterwarnings("ignore")

import pickle

import torch
import torch.nn as nn

from transformers.models.roberta.modeling_roberta import RobertaModel
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer

from LibMTL.config import prepare_args, LibMTL_args
from LibMTL.metrics import AccMetric, AllMetric
from LibMTL.loss import CELoss
from LibMTL.trainer import Trainer
import LibMTL.weighting as weighting_method
import LibMTL.architecture as architecture_method
from LibMTL.utils import set_random_seed

from dataloader_MTLM_PKEN import nlp_dataloader

# from models.BiLSTM import Config, Model
# from models.Att_bilstm.att_bilstm import AttBiLSTM, Config
import utils_MTLM_PKEN as utils

class Config(object):
    """配置参数"""

    def __init__(self):
        self.model_name = 'RoBETRa'
        
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
   
        self.emb_size = 768
        self.dropout = 0.1


    def get_param(self):
        return [self.model_name, self.emb_size, self.dropout]

    def set_attr(self, **args):
        for key, value in args.items():
            if key not in self.__dict__.keys():
                raise IndexError('{}不在配置文件中'.format(key))
            self.__setattr__(key, value)

def parse_args(parser):
    parser.add_argument('--dataset', default='mmh4', type=str, help='office-31, office-home')
    parser.add_argument('--bs', default=2, type=int, help='batch size')
    parser.add_argument('--dataset_path', default='/', type=str, help='dataset path')
    parser.add_argument('--dropout_transformer', default=0.1, type=float, help='dropout_transformer')
    parser.add_argument('--nhead', default=1, type=int, help='dataset path')
    parser.add_argument('--dim_feedforward', default=1024, type=int, help='dataset path')
    parser.add_argument('--num_layers', default=1, type=int, help='dataset path')
    parser.add_argument('--dropout_', default=0.1, type=float, help='dataset path')
    return parser.parse_args()

def main(params, config_encoder):
    kwargs, optim_param, scheduler_param = prepare_args(params)

    if params.dataset == 'mmh':
        task_name = ['mmh']
    elif params.dataset == 'mmh1':
        task_name = ['self_harm', 'adhd', 'anxiety', 'bipolar', 'depression', 'ptsd']
    elif params.dataset == 'mmh2':
        task_name = ['self_harm', 'depression', 'anxiety']
    elif params.dataset == 'mmh3':
        task_name = ['adhd']
    elif params.dataset == 'mmh4':
        task_name = ['self_harm', 'depression']
    elif params.dataset == 'mmh5':
        task_name = ['self_harm', 'bipolar']
    elif params.dataset == 'mmh6':
        task_name = ['self_harm', 'depression_new']
    elif params.dataset == 'mmh7':
        task_name = ['anxiety', 'ptsd']
    elif params.dataset == 'mmh8':
        task_name = ['self_harm', 'bipolar']
    elif params.dataset == 'mmh9':
        task_name = ['bipolar', 'ptsd']
    
    else:
        raise ValueError('No support dataset {}'.format(params.dataset))

    

    # 1. 定义任务集
    # weight_dir = {'adhd': 1, 'anxiety': 1, 'bipolar': 1, 'depression':1, 'ptsd': 1, 'mmh': 1, 'self_harm': 1}
    weight_dir = {'adhd': 1, 'anxiety': 1, 'bipolar': 1, 
                  'depression':1, 'ptsd': 1, 'mmh': 1, 'depression_new': 1,
                  'self_harm': 1, 'bipolar_2378': 1, 'bipolar_3567': 1,
                  'bipolar_4756': 1, 'depression_2378': 1, 'depression_3567': 1, 'depression_4756': 1,
                  'ptsd_2378': 1, 'ptsd_3567': 1, 'ptsd_4756': 1,
                  'adhd_2378': 1, 'adhd_3567': 1, 'adhd_4756': 1}
    task_dict = {task: {'metrics': ['Acc'],
                        'metrics_fn': AllMetric(),
                        'loss_fn': CELoss(),
                        'weight': [weight_dir[task]]} for task in task_name}
    
    # 2. 定义模型共享部分
    class Encoder(nn.Module):
        def __init__(self):
            super(Encoder, self).__init__()
            self.roberta = RobertaModel.from_pretrained('roberta地址')
            self.dropout = nn.Dropout(config_encoder.dropout)
            # self.bilstm = AttBiLSTM(config=config_encoder)
            # self.attbilstm.roberta.requires_grad_(False)    # type: ignore            pass
            
        def forward(self, x):

            text_cls = self.roberta(**x).pooler_output # type: ignore
            text_cls = self.dropout(text_cls)
            # words_per_sentence = x['attention_mask'].sum(dim=1)
            # text_cls = self.bilstm(x)
            return text_cls

    # 3. 定义个任务各自的任务
    out_nums = {'adhd': 2, 'anxiety': 2, 'bipolar': 2, 'depression': 2, 'depression_new': 2,
                'ptsd': 2, 'mmh': 6, 'self_harm': 4,
                'bipolar_2378': 2, 'bipolar_3567': 2, 'bipolar_4756': 2,
                'depression_2378': 2, 'depression_3567': 2, 'depression_4756': 2,
                'ptsd_2378': 2, 'ptsd_3567': 2, 'ptsd_4756': 2,
                'adhd_2378': 2, 'adhd_3567': 2, 'adhd_4756': 2}
    # decoders = nn.ModuleDict({task: nn.Linear(768, out_nums[task]) for task in list(task_dict.keys())})
    decoders = nn.ModuleDict({task:  nn.Sequential(nn.Linear(768 * 3, 768), nn.ReLU(), nn.Linear(768, out_nums[task]))  for task in list(task_dict.keys())})
    # 4. 定义多任务学习模型
    MentalIllnessModel = Trainer(task_dict=task_dict, 
                                    weighting=weighting_method.__dict__[params.weighting], 
                                    architecture=architecture_method.__dict__[params.arch], 
                                    encoder_class=Encoder, 
                                    decoders=decoders,
                                    rep_grad=params.rep_grad,
                                    multi_input=params.multi_input,
                                    optim_param=optim_param,
                                    scheduler_param=scheduler_param,
                                    **kwargs)

    # dataloder设置
    tokenizer = RobertaTokenizer.from_pretrained('roberta地址')
    data_loader, _ = nlp_dataloader(dataset=params.dataset, 
                                        tokenizer=tokenizer,
                                        batch_size=params.bs,
                                        pad_size=512, 
                                        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    train_dataloaders = {task: data_loader[task]['train'] for task in task_name}
    val_dataloaders = {task: data_loader[task]['val'] for task in task_name}
    test_dataloaders = {task: data_loader[task]['test'] for task in task_name}
    # test_data_6 = nlp_dataloader_6(tokenizer=tokenizer,
    #                                 batch_size=params.bs,
    #                                 pad_size=512, 
    #                                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))                                                                              
    # 5. 训练模型
    weights, _ = MentalIllnessModel.train(train_dataloaders=train_dataloaders, 
                      test_dataloaders=test_dataloaders, 
                      epochs=25,
                      return_weight=True)
    
    cur_time, param_model = utils.get_global()
    utils.record_weight(weights, task_name, cur_time, param_model)


def test_train(params, config_encoder, **kwargs):
    
    utils.init_()
    

    utils.set_global(lr=kwargs['lr'],
                     batchsize=kwargs['batchsize'], 
                     config=config_encoder)
    

    _, param_model = utils.get_global()

    # params.dataset = task
    # params.bs = batchsize
    # params.lr = lr
    params.weighting = 'EW'
    params.arch = 'MTLM_PKEN'
    params.seed = 88
    
    ## MTLM_PKEN
    params.img_size = (1, 768)
    params.num_experts = (1, 1, 1)

    # params.dropout_transformer = dropout_transformer
    # params.dim_feedforward = dim_feedforward
    # params.num_layers = num_layers
    # params.nhead = nhead
    # params.lr = 0.0015
    # # set device
    # set_device(params.gpu_id)
    # set random seed
    set_random_seed(params.seed)
    main(params, config_encoder)


if __name__ == "__main__":
    params = parse_args(LibMTL_args)
    config_encoder = Config()
    model_param_dict =  {'emb_size': 768,  
                        'dropout': params.dropout_}
    config_encoder.set_attr(**model_param_dict)
    # params.bs = 6
    # params.lr = 1.5e-5
    # params.dataset = 'mmh4'
    # params.bs = 12
    # task='mmh4', dropout_transformer = 0.1, dim_feedforward = 512, num_layers = 1, nhead = 1 , task='mmh4', num_layers=2, nhead=4, dim_feedforward=256 lr=1.5e-5, batchsize=6
    test_train(params, config_encoder, lr=params.lr, batchsize=params.bs)
