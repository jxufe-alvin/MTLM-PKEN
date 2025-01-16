import pickle
import os
import time
import copy
import re

def init_():
    global TimeStr
    global LR
    global BATCHSIZE
    global CONFIG
    # global PAD_SIZE  # 每句话处理成的长度(短填长切)
    # global EMBED  # 字向量维度，原实现为300
    # global DIM_MODEL  # hidden size,原参数为300，一般设置为64*num_head
    # global HIDDEN # unsed
    # global LAST_HADDEN  # unsed
    # global NUM_HEAD  # 原参数为5，一般取8,12,24
    # global NUM_ENCODER  # 原参数为2，一般取12,24

    

def set_global(lr, batchsize, config):
    global TimeStr
    global LR
    global BATCHSIZE
    global CONFIG
    # global PAD_SIZE  # 每句话处理成的长度(短填长切)
    # global EMBED  # 字向量维度，原实现为300
    # global DIM_MODEL  # hidden size,原参数为300，一般设置为64*num_head
    # global HIDDEN # unsed
    # global LAST_HADDEN  # unsed
    # global NUM_HEAD  # 原参数为5，一般取8,12,24
    # global NUM_ENCODER  # 原参数为2，一般取12,24



    TimeStr = get_time()
    LR = lr
    BATCHSIZE = batchsize
    CONFIG = config
    # PAD_SIZE = pad_size  # 每句话处理成的长度(短填长切)
    # EMBED =embed  # 字向量维度，原实现为300
    # DIM_MODEL = dim_model # hidden size,原参数为300，一般设置为64*num_head
    # HIDDEN = hidden # unsed
    # LAST_HADDEN =last_hidden # unsed
    # NUM_HEAD = num_head # 原参数为5，一般取8,12,24
    # NUM_ENCODER = num_encoder # 原参数为2，一般取12,24

def get_global():
    param_ = CONFIG.get_param()
    param_list = [LR, BATCHSIZE]
    param_list.extend(CONFIG.get_param())
    return TimeStr, param_list


def get_time():
    return time.strftime('%Y-%m-%d-%H',time.localtime())

def record_F1(epoch, results, task_name, cur_time, params_model):
    
    if epoch==0 and not os.path.exists('./results_MTLM_PKEN/result_roberta/{}'.format('_'.join(task_name))):
        os.makedirs('./results_MTLM_PKEN/result_roberta/{}'.format('_'.join(task_name)))
        # cur_time = time.strftime('%Y-%m-%d-%H',time.localtime())

    # with open('./results_MTLM_PKEN/result_roberta/{}/{}_{}.txt'.format('_'.join(task_name), cur_time,
    # 
    #                                            '_'.join(params_model)), 'a+') as f:
    re_find_num = re.compile(r'\d+[.]*\d+')

    data_dir = {}
    for key, value in results.items():
        data_dir['epoch'] = epoch
        data_dir[key + '_acc'] = value[0]
        macro_F1_str =value[2].split('\n')[-3]
        data_dir[key + '_macro_F1'] = re_find_num.findall(macro_F1_str)[-2]
        weighted_F1_str = value[2].split('\n')[-2]
        data_dir[key + '_weighted_F1'] = re_find_num.findall(weighted_F1_str)[-2]
        data_dir[key + '_confusion_matrix'] = str(value[3])


    params_model = [str(i) for i in params_model]
    if not os.path.exists('./results_MTLM_PKEN/result_roberta/{}/aaaa_{}_{}.txt'.format('_'.join(task_name), cur_time, '_'.join(params_model))):
        with open('./results_MTLM_PKEN/result_roberta/{}/aaaa_{}_{}.txt'.format('_'.join(task_name), cur_time, '_'.join(params_model)), 'w') as f:
            f.write('\t'.join(data_dir.keys()))
            f.write('\t\n')

    with open('./results_MTLM_PKEN/result_roberta/{}/aaaa_{}_{}.txt'.format('_'.join(task_name), cur_time, '_'.join(params_model)), 'a+') as f:
        for key, value in data_dir.items():
            if 'confusion_matrix' in key:
                f.write('{}\t'.format(value.replace('\n ', '-')))
            else:
                f.write('{}\t'.format(value))
        f.write('\n')

    # with open('./result_594/{}/{}_{}_{}.pkl'.format('_'.join(task_name), cur_time, epoch, '_'.join(params_model)), 'wb') as f:
    #     pickle.dump(results, f)


def record_loss_and_acc(epoch, mode, results, task_name, loss_item, end_time, beg_time, cur_time, params_model, is_val = False):

    if epoch==0 and not os.path.exists('./losss_MTLM_PKEN/loss_roberta/{}'.format('_'.join(task_name))):
        os.makedirs('./losss_MTLM_PKEN/loss_roberta/{}'.format('_'.join(task_name)))
    params_model = [str(i) for i in params_model]

    if not os.path.exists('./losss_MTLM_PKEN/loss_roberta/{}/aaa_{}_{}.txt'.format('_'.join(task_name), cur_time, '_'.join(params_model))):
        with open('./losss_MTLM_PKEN/loss_roberta/{}/aaa_{}_{}.txt'.format('_'.join(task_name), cur_time, '_'.join(params_model)), 'w') as f:
            f.write('epoch')
            for tn, task in enumerate(task_name):
                f.write('\t{}_train_loss'.format(task))
                f.write('\t{}_train_acc'.format(task))
            for tn, task in enumerate(task_name):
                f.write('\t{}_test_loss'.format(task))
                f.write('\t{}_test_acc'.format(task))
            f.write('\n')

    f = open('./losss_MTLM_PKEN/loss_roberta/{}/{}_{}.txt'.format('_'.join(task_name), cur_time, '_'.join(params_model)), 'a+')
    f2 = open('./losss_MTLM_PKEN/loss_roberta/{}/aaa_{}_{}.txt'.format('_'.join(task_name), cur_time, '_'.join(params_model)), 'a+')
    if mode == 'train':
        f.write('Epoch: {:04d} | '.format(epoch))
        f2.write('{:04d}'.format(epoch))
    if mode == 'train':
        p_mode = 'TRAIN'
    elif mode == 'val':
        p_mode = 'VAL'
    else:
        p_mode = 'TEST'
    f.write('{}: '.format(p_mode))
    for tn, task in enumerate(task_name):
        f.write('{:.4f} '.format(loss_item[tn]))
        f.write('{:.4f} '.format(results[task][0]))
        f.write('| ')
        f2.write('\t{:.4f}'.format(loss_item[tn]))
        f2.write('\t{:.4f}'.format(results[task][0]))
    f.write('Time: {:.4f}'.format(end_time-beg_time))
    f.write(' | ') if mode!='test' else f.write('\n')
    if mode == 'test':
        f2.write('\n')
    f.close()
    f2.close()


def record_weight(weights, task_name, cur_time, params_model):
    
    if not os.path.exists('./weights_MTLM_PKEN/weights_roberta/{}'.format('_'.join(task_name))):
        os.makedirs('./weights_MTLM_PKEN/weights_roberta/{}'.format('_'.join(task_name)))
        # cur_time = time.strftime('%Y-%m-%d-%H',time.localtime())
    params_model = [str(i) for i in params_model]
    with open('./weights_MTLM_PKEN/weights_roberta/{}/{}_{}.pkl'.format('_'.join(task_name), cur_time, '_'.join(params_model)), 'wb') as f:
        pickle.dump(weights, f)