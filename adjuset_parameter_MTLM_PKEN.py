from train_mm_MTLM_PKEN import parse_args, Config, test_train
from LibMTL.config import LibMTL_args

if __name__ == "__main__":
    params = parse_args(LibMTL_args)
    config_encoder = Config()
    # for task in ['mmh5', 'mmh10', 'mmh11', 'mmh12', 'mmh13', 'mmh14']:
    for task in ['mmh4']:
        for nhead in [5]:
            for num_layers in [2]:
                for dim_feedforward in [256]:   # [128, 256, 512, 1024]
                    try:
                        with open('./logs/log_2024_1_5.txt', 'a+') as f:
                            f.write('MTLM_PKEN——EW——dropout:{}\t{}开始训练\n'.format( nhead, task))
                        
                        model_param_dict =  {'emb_size': 768,  
                                            'dropout': 0.1}
                        config_encoder.set_attr(**model_param_dict)

                        test_train(params, config_encoder, lr=1.5e-5, batchsize=6, task=task, num_layers=num_layers, nhead=nhead, dim_feedforward=dim_feedforward)

                        with open('./logs/log_2024_1_5.txt', 'a+') as f:
                            f.write('MTLM_PKEN——EW——dropout:{}\t{}结束训练\n'.format( nhead, task))

                    except Exception as e:

                        with open('./logs/log_2024_1_5.txt', 'a+') as f:
                            f.write('MTLM_PKEN——EW——dropout:{}\t{}未完成训练，原因是{}\n'.format(nhead, task, str(e)))
                        continue


# scp -r  pengwenzhong@172.29.42.100:/share/home/pengwenzhong/zzf/xc/LibMTL_MultiMental_MTLM_PKEN
