import pickle
import os
import time
import copy
import re

def init_():
    global TimeStr
    global LR
    global RNN_SIZE
    global RNN_LAYERS
    global DROUPOUT

    

def set_global(lr, rnn_size, rnn_layers, droutout, batchsize):
    global TimeStr
    global LR
    global RNN_SIZE
    global RNN_LAYERS
    global DROUPOUT
    global BATCHSIZE
    
    TimeStr = get_time()
    LR = lr
    RNN_SIZE = rnn_size
    RNN_LAYERS = rnn_layers
    DROUPOUT = droutout
    BATCHSIZE = batchsize

def get_global():
    return TimeStr, [LR , RNN_SIZE, RNN_LAYERS, DROUPOUT, BATCHSIZE]


def get_time():
    return time.strftime('%Y-%m-%d-%H',time.localtime())

def record_F1(epoch, results, task_name, cur_time, params_model):
    
    if epoch==0 and not os.path.exists('./results/result_bi_lstm/{}'.format('_'.join(task_name))):
        os.makedirs('./results/result_bi_lstm/{}'.format('_'.join(task_name)))
        # cur_time = time.strftime('%Y-%m-%d-%H',time.localtime())

    # with open('./results/result_bi_lstm/{}/{}_{}.txt'.format('_'.join(task_name), cur_time,
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
    if not os.path.exists('./results/result_bi_lstm/{}/aaaa_{}_{}.txt'.format('_'.join(task_name), cur_time, '_'.join(params_model))):
        with open('./results/result_bi_lstm/{}/aaaa_{}_{}.txt'.format('_'.join(task_name), cur_time, '_'.join(params_model)), 'w') as f:
            f.write('\t'.join(data_dir.keys()))
            f.write('\t\n')

    with open('./results/result_bi_lstm/{}/aaaa_{}_{}.txt'.format('_'.join(task_name), cur_time, '_'.join(params_model)), 'a+') as f:
        for key, value in data_dir.items():
            if 'confusion_matrix' in key:
                f.write('{}\t'.format(value.replace('\n ', '-')))
            else:
                f.write('{}\t'.format(value))
        f.write('\n')

    # with open('./result_594/{}/{}_{}_{}.pkl'.format('_'.join(task_name), cur_time, epoch, '_'.join(params_model)), 'wb') as f:
    #     pickle.dump(results, f)


def record_loss_and_acc(epoch, mode, results, task_name, loss_item, end_time, beg_time, cur_time, params_model, is_val = False):

    if epoch==0 and not os.path.exists('./losss/loss_bi_lstm/{}'.format('_'.join(task_name))):
        os.makedirs('./losss/loss_bi_lstm/{}'.format('_'.join(task_name)))
    params_model = [str(i) for i in params_model]

    if not os.path.exists('./losss/loss_bi_lstm/{}/aaa_{}_{}.txt'.format('_'.join(task_name), cur_time, '_'.join(params_model))):
        with open('./losss/loss_bi_lstm/{}/aaa_{}_{}.txt'.format('_'.join(task_name), cur_time, '_'.join(params_model)), 'w') as f:
            f.write('epoch')
            for tn, task in enumerate(task_name):
                f.write('\t{}_train_loss'.format(task))
                f.write('\t{}_train_acc'.format(task))
            for tn, task in enumerate(task_name):
                f.write('\t{}_test_loss'.format(task))
                f.write('\t{}_test_acc'.format(task))
            f.write('\n')

    f = open('./losss/loss_bi_lstm/{}/{}_{}.txt'.format('_'.join(task_name), cur_time, '_'.join(params_model)), 'a+')
    f2 = open('./losss/loss_bi_lstm/{}/aaa_{}_{}.txt'.format('_'.join(task_name), cur_time, '_'.join(params_model)), 'a+')
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
    
    if not os.path.exists('./weights/weights_bi_lstm/{}'.format('_'.join(task_name))):
        os.makedirs('./weights/weights_bi_lstm/{}'.format('_'.join(task_name)))
        # cur_time = time.strftime('%Y-%m-%d-%H',time.localtime())
    params_model = [str(i) for i in params_model]
    with open('./weights/weights_bi_lstm/{}/{}_{}.pkl'.format('_'.join(task_name), cur_time, '_'.join(params_model)), 'wb') as f:
        pickle.dump(weights, f)