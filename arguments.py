import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', help='If training is to be done on a GPU')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Name of the dataset used.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size used for training and testing')
    parser.add_argument('--train_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--latent_dim', type=int, default=32, help='The dimensionality of the VAE latent dimension')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to where the data is')
    parser.add_argument('--beta', type=float, default=1, help='Hyperparameter for training. The parameter for VAE')
    parser.add_argument('--num_adv_steps', type=int, default=1, help='Number of adversary steps taken for every task model step')
    parser.add_argument('--num_vae_steps', type=int, default=2, help='Number of VAE steps taken for every task model step')
    parser.add_argument('--adversary_param', type=float, default=1, help='Hyperparameter for training. lambda2 in the paper')
    parser.add_argument('--out_path', type=str, default='./results', help='Path to where the output log will be')
    parser.add_argument('--log_name', type=str, default='accuracies.log', help='Final performance of the models will be saved with this name')
    parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument("--gpu", type=int, default=0,help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=300,
            help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=50,
            help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
            help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
            help="Weight for L2 loss")
    parser.add_argument("--self-loop", action='store_true',default=True,
            help="graph self-loop (default=False)")
    parser.add_argument('--xi', type=float, default=1, metavar='XI',
            help='hyperparameter of VAT (default: 0.1)')
    parser.add_argument('--eps', type=float, default=0.02, metavar='EPS',
            help='hyperparameter of VAT (default: 1.0)')
    parser.add_argument('--ip', type=int, default=1, metavar='IP',
            help='hyperparameter of VAT (default: 1)')
    args = parser.parse_args()
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    
    return args
