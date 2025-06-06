def parse_args():
    # parser = argparse.ArgumentParser()
    # # mode
    # parser.add_argument('--zero_shot', action='store_true')
    # # data
    # parser.add_argument('--data_name', type=str, required=True, choices=['inspired', 'redial', 'pearl'])
    # # hyperparameter
    # parser.add_argument('--alpha', type=float, default=0.5)
    # parser.add_argument('--beta', type=float, default=0.2)
    # parser.add_argument('--temp', type=float, default=0.05)
    # parser.add_argument("--query_used_info", type=str, nargs='+', choices=['c', 'l', 'd'], default=['c', 'l', 'd'])
    # parser.add_argument("--doc_used_info", type=str, nargs='+', choices=['m', 'pref'], default=['m', 'pref'])
    # # train
    # parser.add_argument("--bf16", action="store_true")
    # parser.add_argument("--negative_sample", type=int, default=16)
    # parser.add_argument('--epochs', type=int, default=100)
    # parser.add_argument('--accumulation_steps', type=int, default=8)
    # parser.add_argument('--base_model_name', type=str, default='nvidia/NV-Embed-v1')
    # parser.add_argument('--learning_rate', type=float, default=1e-4)
    # parser.add_argument('--patience', type=int, default=5)
    # parser.add_argument('--batch_size', type=int, default=10)
    # parser.add_argument('--ckpt_save_path', type=str, default='checkpoints')
    # parser.add_argument('--seed', type=int, default=2024)
    # parser.add_argument('--cutoff', type=int, nargs='+', default=[5, 10, 50])
    # # wandb
    # parser.add_argument("--wandb_project", type=str)
    # parser.add_argument("--wandb_entity", type=str)
    # parser.add_argument('--wandb_group', type=str)

    # args = parser.parse_args()

    args = {
        # mode
        'zero_shot': False,

        # data
        'data_name': 'inspired',

        # hyperparameter
        'alpha': 0.5,
        'beta': 0.2,
        'temp': 0.05,
        'query_used_info': ['c', 'l', 'd'],
        'doc_used_info': ['m', 'pref'],

        # train
        'bf16': True,
        'negative_sample': 16,
        'epochs': 100,
        'accumulation_steps': 8,
        'base_model_name': 'nvidia/NV-Embed-v1',
        'learning_rate': 1e-4,
        'patience': 5,
        'batch_size': 10,
        'ckpt_save_path': 'checkpoints',
        'seed': 2024,
        'cutoff': [5, 10, 50],

        # wandb
        'wandb_project': None,
        'wandb_entity': None,
        'wandb_group': None,
    }


    return args


if __name__ == '__main__':
    args = parse_args()
    print(type(args))
    print(vars(args))