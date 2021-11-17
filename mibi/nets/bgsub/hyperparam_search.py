import os


def generate_hyperparam_file(learning_rates=[1e-3],
                             weight_decays=[0, 1],
                             energy_coefs=[1e-3, 1],
                             epochs=[10],
                             batch_size=30,
                             num_runs=5,
                             param_prefix='hyperparam_search',
                             base_dir='/home/mschachter/CHOP_TMA_pilot',
                             base_cmd='python mibi/nets/bgsub/train.py'):

    all_params = list()

    chan_file = os.path.join(base_dir, 'info', 'channels.csv')
    img_dir = os.path.join(base_dir, 'extracted')

    for learning_rate in learning_rates:
        for weight_decay in weight_decays:
            for epoch in epochs:
                for energy_coef in energy_coefs:
                    all_params.append({'weight_decay':f"{weight_decay:.6f}",
                                        'learning_rate':f"{learning_rate:.6f}",
                                        'epochs':f"{epoch}",
                                        'images_dir':img_dir,
                                        'channel_file':chan_file,
                                        'batch_size':batch_size,
                                        'energy_coef':energy_coef
                                        })

    cmds = list()
    for k,params in enumerate(all_params):
        for j in range(num_runs):
            model_desc = f'{param_prefix}_{k}_{j}'
            all_strs = [base_cmd]
            all_strs.extend([f'--{p} {v}' for p,v in params.items()])
            all_strs.append(f'--model_desc {model_desc}')
            cmds.append(' '.join(all_strs))

    print('# of total jobs: ', len(cmds))

    with open('hyperparam_search.sh', 'w') as f:
        f.write('\n'.join(cmds))

if __name__ == '__main__':
    generate_hyperparam_file(param_prefix='hyperparam_logcos_search')




