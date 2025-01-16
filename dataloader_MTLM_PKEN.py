import torch
from torch.utils.data import Dataset, DataLoader

import pickle
import pandas as pd
import numpy as np

import copy


class multi_mental_health(Dataset):
    def __init__(self, data_path):
        super().__init__()
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        self.posts = self.data.post.values.tolist()  # type: ignore
        self.labels = self.data.class_id.values.tolist()  # type: ignore
        self.posts_embedding = self.data.embedding.values.tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        post = self.posts[index]
        label = self.labels[index]
        embedding_re = self.posts_embedding[index]

        return post, label, embedding_re

def building_dataloader_mental_health(train_path, test_path, tokenizer, batch_size, pad_size, device, add_title):
    """构建一个数据集迭代器，

    Args:
        config (class): 配置参数的实例
    """

    def collate_fn(data):
        """怎么取数据

        Args:
            data (dataset): 上面构建的数据集

        Returns:
            _type_: _description_
        """
        posts = [i[0] for i in data]
        labels = [i[1] for i in data]
        embeddings = [i[2] for i in  data]

        #编码
        inputs = tokenizer.batch_encode_plus(batch_text_or_text_pairs=posts,
                                    truncation=True,
                                    padding='max_length',
                                    max_length=pad_size,   #   修改过
                                    return_tensors='pt')
                                    # return_length=True)

        for  i in inputs:
            inputs[i] = inputs[i].to(device)

        embeddings = torch.from_numpy(np.stack(embeddings)).squeeze(1)

        labels = torch.LongTensor(labels).to(device)
        #input_ids:编码之后的数字
        #attention_mask:是补零的位置是0,其他位置是1
        # input_ids = data['input_ids'].to(device)
        # attention_mask = data['attention_mask'].to(device)
        # if model_name == 'bert-base-uncased':
        #     token_type_ids = data['token_type_ids'].to(device)

        return [inputs, embeddings], labels

    dataset_train = multi_mental_health(train_path)
    dataset_test = multi_mental_health(test_path)

    train_loader = DataLoader(dataset=dataset_train,
                                    batch_size=batch_size,
                                    collate_fn=collate_fn,
                                    shuffle=True,
                                    drop_last=True)
    test_loader = DataLoader(dataset=dataset_test,
                                    batch_size=batch_size,
                                    collate_fn=collate_fn,
                                    shuffle=True,
                                    drop_last=True)

    val_loader = copy.deepcopy(test_loader)

    return train_loader, val_loader, test_loader



def nlp_dataloader(dataset, tokenizer, batch_size, pad_size, device):
    """_summary_

    Args:
        data_path (dir): {'task1': datapath, 'task2': datapath2}
        model_name (_type_): 使用的特征提取器的模型结构
        batch_size (_type_): batchsize
        word_pad_len (_type_): _description_
        device (_type_): _description_

    Returns:
        _type_: _description_
    """
    if dataset == 'CLP_ERISK':
        tasks = ['erisk', 'clp2019']
    elif dataset == 'ERISK':
        tasks = ['erisk']
    elif dataset == 'mmh':
        tasks = ['mmh']
    elif dataset == 'mmh1':
        tasks = ['self_harm', 'adhd', 'anxiety', 'bipolar', 'depression', 'ptsd']
    elif dataset == 'mmh2':
        tasks = ['self_harm', 'depression', 'anxiety'] 
    elif dataset == 'mmh3':
        tasks = ['adhd']
    elif dataset == 'mmh4':
        tasks = ['self_harm', 'depression']
    elif dataset == 'mmh5':
        tasks = ['self_harm', 'bipolar']
    elif dataset == 'mmh6':
        tasks = ['self_harm', 'depression_new']
    elif dataset == 'mmh7':
        tasks = ['anxiety', 'ptsd']
    elif dataset == 'mmh8':
        tasks = ['self_harm', 'bipolar']
    elif dataset == 'mmh9':
        tasks = ['bipolar', 'ptsd']
    
    else:
        raise ValueError('No support dataset {}'.format(dataset))
    data_loader = {}
    iter_data_loader = {}
    for k, d in enumerate(tasks):
        if d == 'mmh':
            data_loader[d] = {}
            iter_data_loader[d] = {}
            _train_path = './dataset_embedded_word_word/both_train.pkl'
            _test_path = './dataset_embedded_word/both_test.pkl'
            data_loader[d]['train'], data_loader[d]['val'], data_loader[d]['test'] = building_dataloader_mental_health(train_path=_train_path, 
                                                                                                                        test_path=_test_path, 
                                                                                                                        batch_size=batch_size,
                                                                                                                        device=device)
            iter_data_loader[d]['train'] = iter(data_loader[d]['train'])
            iter_data_loader[d]['test'] = iter(data_loader[d]['test'])
            iter_data_loader[d]['val'] = iter(data_loader[d]['val'])
        elif 'self_harm' in d:
            data_loader[d] = {}
            iter_data_loader[d] = {}
            _train_path = f'./dataset_embedding/train_{d}.pkl'
            _test_path = f'./dataset_embedding/test_self_harm.pkl'
            data_loader[d]['train'], data_loader[d]['val'], data_loader[d]['test'] = building_dataloader_mental_health(train_path=_train_path, 
                                                                                                                        test_path=_test_path, 
                                                                                                                        tokenizer=tokenizer,
                                                                                                                        batch_size=batch_size,
                                                                                                                        pad_size=pad_size,
                                                                                                                        device=device,
                                                                                                                        add_title=False)
            iter_data_loader[d]['train'] = iter(data_loader[d]['train'])
            iter_data_loader[d]['test'] = iter(data_loader[d]['test'])
            iter_data_loader[d]['val'] = iter(data_loader[d]['val'])
        
        # elif 'depression_new' == d:
        #     data_loader[d] = {}
        #     iter_data_loader[d] = {}
        #     _train_path = f'./dataset_embedding/train_{d}.pkl'
        #     _test_path = f'./dataset_embedding/test_depression_new.pkl'
        #     data_loader[d]['train'], data_loader[d]['val'], data_loader[d]['test'] = building_dataloader_mental_health(train_path=_train_path, 
        #                                                                                                                 test_path=_test_path, 
        #                                                                                                                 tokenizer=tokenizer,
        #                                                                                                                 batch_size=batch_size,
        #                                                                                                                 pad_size=pad_size,
        #                                                                                                                 device=device,
        #                                                                                                                 add_title=False)
        #     iter_data_loader[d]['train'] = iter(data_loader[d]['train'])
        #     iter_data_loader[d]['test'] = iter(data_loader[d]['test'])
        #     iter_data_loader[d]['val'] = iter(data_loader[d]['val'])

        elif 'depression' in d:
            data_loader[d] = {}
            iter_data_loader[d] = {}
            _train_path = f'./dataset_embedding/train_{d}.pkl'
            _test_path = f'./dataset_embedding/test_depression.pkl'
            data_loader[d]['train'], data_loader[d]['val'], data_loader[d]['test'] = building_dataloader_mental_health(train_path=_train_path, 
                                                                                                                        test_path=_test_path, 
                                                                                                                        tokenizer=tokenizer,
                                                                                                                        batch_size=batch_size,
                                                                                                                        pad_size=pad_size,
                                                                                                                        device=device,
                                                                                                                        add_title=False)
            iter_data_loader[d]['train'] = iter(data_loader[d]['train'])
            iter_data_loader[d]['test'] = iter(data_loader[d]['test'])
            iter_data_loader[d]['val'] = iter(data_loader[d]['val'])
        elif 'anxiety' in d:
            data_loader[d] = {}
            iter_data_loader[d] = {}
            _train_path = f'./dataset_embedding/train_{d}.pkl'
            _test_path = f'./dataset_embedding/test_anxiety.pkl'
            data_loader[d]['train'], data_loader[d]['val'], data_loader[d]['test'] = building_dataloader_mental_health(train_path=_train_path, 
                                                                                                                        test_path=_test_path, 
                                                                                                                        tokenizer=tokenizer,
                                                                                                                        batch_size=batch_size,
                                                                                                                        pad_size=pad_size,
                                                                                                                        device=device,
                                                                                                                        add_title=False)
            iter_data_loader[d]['train'] = iter(data_loader[d]['train'])
            iter_data_loader[d]['test'] = iter(data_loader[d]['test'])
            iter_data_loader[d]['val'] = iter(data_loader[d]['val'])
        elif 'bipolar' in d:
            data_loader[d] = {}
            iter_data_loader[d] = {}
            _train_path = f'./dataset_embedding/train_{d}.pkl'
            _test_path = f'./dataset_embedding/test_bipolar.pkl'
            data_loader[d]['train'], data_loader[d]['val'], data_loader[d]['test'] = building_dataloader_mental_health(train_path=_train_path, 
                                                                                                                        test_path=_test_path, 
                                                                                                                        tokenizer=tokenizer,
                                                                                                                        batch_size=batch_size,
                                                                                                                        pad_size=pad_size,
                                                                                                                        device=device,
                                                                                                                        add_title=False)
            iter_data_loader[d]['train'] = iter(data_loader[d]['train'])
            iter_data_loader[d]['test'] = iter(data_loader[d]['test'])
            iter_data_loader[d]['val'] = iter(data_loader[d]['val'])
        elif 'ptsd' in d:
            data_loader[d] = {}
            iter_data_loader[d] = {}
            _train_path = f'./dataset_embedding/train_{d}.pkl'
            _test_path = f'./dataset_embedding/test_ptsd.pkl'
            data_loader[d]['train'], data_loader[d]['val'], data_loader[d]['test'] = building_dataloader_mental_health(train_path=_train_path, 
                                                                                                                        test_path=_test_path, 
                                                                                                                        tokenizer=tokenizer,
                                                                                                                        batch_size=batch_size,
                                                                                                                        pad_size=pad_size,
                                                                                                                        device=device,
                                                                                                                        add_title=False)
            iter_data_loader[d]['train'] = iter(data_loader[d]['train'])
            iter_data_loader[d]['test'] = iter(data_loader[d]['test'])
            iter_data_loader[d]['val'] = iter(data_loader[d]['val'])
        elif 'adhd' in d:
            data_loader[d] = {}
            iter_data_loader[d] = {}
            _train_path = f'./dataset_embedded_word/train_{d}.pkl'
            _test_path = f'./dataset_embedded_word/test_adhd.pkl'
            data_loader[d]['train'], data_loader[d]['val'], data_loader[d]['test'] = building_dataloader_mental_health(train_path=_train_path, 
                                                                                                                        test_path=_test_path, 
                                                                                                                        tokenizer=tokenizer,
                                                                                                                        batch_size=batch_size,
                                                                                                                        pad_size=pad_size,
                                                                                                                        device=device,
                                                                                                                        add_title=False)
        else:
            data_loader[d] = {}
            iter_data_loader[d] = {}
            _train_path = f'./dataset_embedded_word/train_{d}.pkl'
            _test_path = f'./dataset_embedded_word/test_{d}.pkl'
            data_loader[d]['train'], data_loader[d]['val'], data_loader[d]['test'] = building_dataloader_mental_health(train_path=_train_path, 
                                                                                                                            test_path=_test_path, 
                                                                                                                            batch_size=batch_size,
                                                                                                                            device=device)
            iter_data_loader[d]['train'] = iter(data_loader[d]['train'])
            iter_data_loader[d]['test'] = iter(data_loader[d]['test'])
            iter_data_loader[d]['val'] = iter(data_loader[d]['val'])

    
        
    return data_loader, iter_data_loader


# def nlp_dataloader_6( tokenizer, batch_size, pad_size, device):
#     """_summary_

#     Args:
#         data_path (dir): {'task1': datapath, 'task2': datapath2}
#         model_name (_type_): 使用的特征提取器的模型结构
#         batch_size (_type_): batchsize
#         word_pad_len (_type_): _description_
#         device (_type_): _description_

#     Returns:
#         _type_: _description_
#     """
#     _train_path = './dataset1/both_train.csv'
#     _test_path = './dataset1/both_test.csv'
#     _, _, test_dataloader_6 = building_dataloader_mental_health(train_path=_train_path, 
#                                                                     test_path=_test_path, 
#                                                                     batch_size=batch_size,
#                                                                     device=device)
    
#     iter_test_dataloader_6 = iter(test_dataloader_6)
        
    
# ./LibMTL_MultiMental_Baseline/LibMTL_MultiMental_sampled/dataset_embedded_word      
#     return test_dataloader_6, iter_test_dataloader_6
