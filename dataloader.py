import torch
from torch.utils.data import Dataset, DataLoader

import pandas as pd

import copy


class multi_mental_health(Dataset):
    def __init__(self, data_path, add_title=True):
        super().__init__()
        self.data = pd.read_csv(data_path)
        self.posts = self.data.post.values.tolist()  # type: ignore
        if add_title:
            self.data['title_post'] = self.data.apply(lambda x: x['title'] + x['post'], axis=1)
            self.posts = self.data.title_post.values.tolist()  # type: ignore
        else:
            self.posts = self.data.post.values.tolist()
        self.labels = self.data.class_id.values.tolist()  # type: ignore

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        post = self.posts[index]
        label = self.labels[index]

        return post, label

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

        #编码
        inputs = tokenizer.batch_encode_plus(batch_text_or_text_pairs=posts,
                                    truncation=True,
                                    padding='max_length',
                                    max_length=pad_size,   #   修改过
                                    return_tensors='pt')
                                    # return_length=True)

        for  i in inputs:
            inputs[i] = inputs[i].to(device)

        labels = torch.LongTensor(labels).to(device)
        #input_ids:编码之后的数字
        #attention_mask:是补零的位置是0,其他位置是1
        # input_ids = data['input_ids'].to(device)
        # attention_mask = data['attention_mask'].to(device)
        # if model_name == 'bert-base-uncased':
        #     token_type_ids = data['token_type_ids'].to(device)

        return inputs, labels

    dataset_train = multi_mental_health(train_path, add_title)
    dataset_test = multi_mental_health(test_path, add_title)

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
        tasks = ['self_harm']
    elif dataset == 'mmh6':
        tasks = ['self_harm', 'anxiety']
    elif dataset == 'mmh7':
        tasks = ['self_harm', 'adhd']
    elif dataset == 'mmh8':
        tasks = ['self_harm', 'bipolar']
    elif dataset == 'mmh9':
        tasks = ['self_harm', 'ptsd']
    elif dataset == 'mmh10':
        tasks = ['adhd']
    elif dataset == 'mmh11':
        tasks = ['bipolar']
    elif dataset == 'mmh12':
        tasks = ['ptsd']
    elif dataset == 'mmh13':
        tasks = ['depression']
    elif dataset == 'mmh14':
        tasks = ['anxiety']
    elif dataset == 'mmh15':
        tasks = ['anxiety', 'depression']
    elif dataset == 'mmh16':
        tasks = ['anxiety', 'ptsd']
    elif dataset == 'mmh17':
        tasks = ['anxiety', 'bipolar']
    elif dataset == 'mmh18':
        tasks = ['anxiety', 'adhd']
    elif dataset == 'mmh19':
        tasks = ['depression', 'ptsd']
    elif dataset == 'mmh20':
        tasks = ['depression', 'bipolar']
    elif dataset == 'mmh21':
        tasks = ['depression', 'adhd']
    elif dataset == 'mmh22':
        tasks = ['ptsd', 'bipolar']
    elif dataset == 'mmh23':
        tasks = ['ptsd', 'adhd']
    elif dataset == 'mmh24':
        tasks = ['bipolar', 'adhd']
    elif dataset == 'mmh25':
        tasks = ['depression', 'anxiety', 'bipolar', 'ptsd', 'adhd']
    elif dataset == 'mmh26':
        tasks = ['anxiety', 'self_harm', 'depression', 'ptsd', 'adhd']
    elif dataset == 'mmh27':
        tasks = ['bipolar', 'depression', 'anxiety', 'ptsd', 'adhd']
    elif dataset == 'mmh28':
        tasks = ['ptsd', 'bipolar', 'adhd']
    elif dataset == 'mmh29':
        tasks = ['adhd', 'self_harm', 'bipolar', 'depression']
    elif dataset == 'mmh30':
        tasks = ['self_harm', 'depression', 'anxiety', 'ptsd', 'adhd']
    elif dataset == 'mmh31':
        tasks = ['depression', 'anxiety', 'bipolar', 'ptsd', 'adhd']
    elif dataset == 'mmh32':
        tasks = ['anxiety', 'self_harm', 'depression', 'bipolar', 'adhd']
    elif dataset == 'mmh33':
        tasks = ['bipolar', 'depression', 'anxiety',  'ptsd', 'adhd']
    elif dataset == 'mmh34':
        tasks = ['ptsd', 'adhd']
    elif dataset == 'mmh35':
        tasks = ['self_harm', 'depression_4756']
    elif dataset == 'mmh36':
        tasks = ['self_harm_297', 'depression']
    elif dataset == 'mmh37':
        tasks = ['self_harm_594', 'depression']
    elif dataset == 'mmh38':
        tasks = ['self_harm_297']
    elif dataset == 'mmh39':
        tasks = ['self_harm_594']
    elif dataset == 'mmh40':
        tasks = ['anxiety', 'bipolar_2378']
    elif dataset == 'mmh41':
        tasks = ['anxiety', 'bipolar_3567']
    elif dataset == 'mmh42':
        tasks = ['anxiety', 'bipolar_4756']
    elif dataset == 'mmh43':
        tasks = ['bipolar', 'depression_2378']
    elif dataset == 'mmh44':
        tasks = ['bipolar', 'depression_3567']
    elif dataset == 'mmh45':
        tasks = ['bipolar', 'depression_4756']

    elif dataset == 'mmh46':
        tasks = ['ptsd', 'bipolar_2378']
    elif dataset == 'mmh47':
        tasks = ['ptsd', 'bipolar_3567']
    elif dataset == 'mmh48':
        tasks = ['ptsd', 'bipolar_4756']

    elif dataset == 'mmh49':
        tasks = ['adhd', 'ptsd_2378']
    elif dataset == 'mmh50':
        tasks = ['adhd', 'ptsd_3567']
    elif dataset == 'mmh51':
        tasks = ['adhd', 'ptsd_4756']
    
    elif dataset == 'mmh52':
        tasks = ['depression', 'adhd_2378']
    elif dataset == 'mmh53':
        tasks = ['depression', 'adhd_3567']
    elif dataset == 'mmh54':
        tasks = ['depression', 'adhd_4756']

    elif dataset == 'mmh55':
        tasks = ['self_harm', 'depression_new']

    elif dataset == 'mmh56':
        tasks = ['depression_new']

    else:
        raise ValueError('No support dataset {}'.format(dataset))
    data_loader = {}
    iter_data_loader = {}
    for k, d in enumerate(tasks):
        if d == 'mmh':
            data_loader[d] = {}
            iter_data_loader[d] = {}
            _train_path = './dataset_used/both_train.csv'
            _test_path = './dataset_used/both_test.csv'
            data_loader[d]['train'], data_loader[d]['val'], data_loader[d]['test'] = building_dataloader_mental_health(train_path=_train_path, 
                                                                                                                        test_path=_test_path, 
                                                                                                                        tokenizer=tokenizer,
                                                                                                                        batch_size=batch_size,
                                                                                                                        pad_size=pad_size,
                                                                                                                        device=device,
                                                                                                                        add_title=True)
            iter_data_loader[d]['train'] = iter(data_loader[d]['train'])
            iter_data_loader[d]['test'] = iter(data_loader[d]['test'])
            iter_data_loader[d]['val'] = iter(data_loader[d]['val'])
        elif 'self_harm' in d:
            data_loader[d] = {}
            iter_data_loader[d] = {}
            _train_path = f'./dataset_used/train_{d}.csv'
            _test_path = f'./dataset_used/test_self_harm.csv'
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
        elif 'depression' in d:
            data_loader[d] = {}
            iter_data_loader[d] = {}
            _train_path = f'./dataset_used/train_{d}.csv'
            _test_path = f'./dataset_used/test_depression.csv'
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
            _train_path = f'./dataset_used/train_{d}.csv'
            _test_path = f'./dataset_used/test_anxiety.csv'
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
            _train_path = f'./dataset_used/train_{d}.csv'
            _test_path = f'./dataset_used/test_bipolar.csv'
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
            _train_path = f'./dataset_used/train_{d}.csv'
            _test_path = f'./dataset_used/test_ptsd.csv'
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
            _train_path = f'./dataset_used/train_{d}.csv'
            _test_path = f'./dataset_used/test_adhd.csv'
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
            _train_path = f'./dataset_used/train_{d}.csv'
            _test_path = f'./dataset_used/test_{d}.csv'
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

    
        
    return data_loader, iter_data_loader


def nlp_dataloader_6( tokenizer, batch_size, pad_size, device):
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
    _train_path = './dataset1/both_train.csv'
    _test_path = './dataset1/both_test.csv'
    _, _, test_dataloader_6 = building_dataloader_mental_health(train_path=_train_path, 
                                                                    test_path=_test_path, 
                                                                    tokenizer=tokenizer,
                                                                    batch_size=batch_size,
                                                                    pad_size=pad_size,
                                                                    device=device,
                                                                    add_title=True)
    
    iter_test_dataloader_6 = iter(test_dataloader_6)
        
    
        
    return test_dataloader_6, iter_test_dataloader_6



