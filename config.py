#-*-coding:utf-8-*-

import argparse
from utils import str2bool


def get_config():
    parser = argparse.ArgumentParser()
    
    # Model configuration.
    parser.add_argument('--mode', type=str, default='train', help='train|test')
    parser.add_argument('--adv_loss_type', type=str, default='rahinge', help='adversarial Loss: ls|original|hinge|rahinge|rals')
    parser.add_argument('--image_size', type=int, default=512, help='image load resolution')
    parser.add_argument('--resize_size', type=int, default=256, help='resolution after resizing')
    parser.add_argument('--test_img_size', type=int, default=512, help='resolution after resizing')
    parser.add_argument('--g_conv_dim', type=int, default=32, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=32, help='number of conv filters in the first layer of D')
    parser.add_argument('--shuffle', type=str, default=True, help='shuffle when load dataset')
    parser.add_argument('--drop_last', type=str2bool, default=False, help=' drop the last incomplete batch')
    parser.add_argument('--version', type=str, default='UEGAN-FiveK', help='UEGAN')
    parser.add_argument('--init_type', type=str, default='orthogonal', help='normal|xavier|kaiming|orthogonal')
    parser.add_argument('--adv_input',type=str2bool, default=True, help='whether discriminator input')
    parser.add_argument('--g_use_sn', type=str2bool, default=False, help='whether use spectral normalization in G')
    parser.add_argument('--d_use_sn', type=str2bool, default=True, help='whether use spectral normalization in D')
    parser.add_argument('--g_act_fun', type=str, default='LeakyReLU', help='LeakyReLU|ReLU|Swish|SELU|none')
    parser.add_argument('--d_act_fun', type=str, default='LeakyReLU', help='LeakyReLU|ReLU|Swish|SELU|none')
    parser.add_argument('--g_norm_fun', type=str, default='none', help='BatchNorm|InstanceNorm|none')
    parser.add_argument('--d_norm_fun', type=str, default='none', help='BatchNorm|InstanceNorm|none')
    
    # Training configuration.
    parser.add_argument('--pretrained_model', type=float, default=0.0)
    parser.add_argument('--total_epochs', type=int, default=100, help='total epochs to update the generator')
    parser.add_argument('--train_batch_size', type=int, default=10, help='mini batch size')
    parser.add_argument('--val_batch_size', type=int, default=1, help='mini batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='subprocesses to use for data loading')
    parser.add_argument('--seed', type=int, default=1990, help='Seed for random number generator')
    parser.add_argument('--g_lr', type=float, default=1e-4, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=4e-4, help='learning rate for D')
    parser.add_argument('--lr_decay', type=str2bool, default=True, help='setup learning rate decay schedule')
    parser.add_argument('--lr_num_epochs_decay', type=int, default=50, help='LambdaLR: epoch at starting learning rate')
    parser.add_argument('--lr_decay_ratio', type=int, default=50, help='LambdaLR: ratio of linearly decay learning rate to zero')
    parser.add_argument('--optimizer_type', type=str, default='adam', help='adam|rmsprop')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--alpha', type=float, default=0.9, help='alpha for rmsprop optimizer')
    parser.add_argument('--lambda_adv', type=float, default=0.10, help='weight for adversarial loss')
    parser.add_argument('--lambda_percep', type=float, default=1.0, help='weight for perceptual loss')
    parser.add_argument('--lambda_idt', type=float, default=0.10, help='weight for identity loss')
    parser.add_argument('--idt_loss_type', type=str, default='l1', help='identity_loss: l1|l2|smoothl1 ')
    parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer, pool_size=0 means no buffer')

    # validation and test configuration
    parser.add_argument('--num_epochs_start_val', type=int, default=8, help='start validate the model')
    parser.add_argument('--val_each_epochs', type=int, default=2, help='validate the model every time after training these epochs')

    # Directories.
    parser.add_argument('--train_img_dir', type=str, default='./data/fivek/train')
    parser.add_argument('--val_img_dir', type=str, default='./data/fivek/val')
    parser.add_argument('--test_img_dir', type=str, default='./data/fivek/test')
    parser.add_argument('--save_root_dir', type=str, default='./results')
    parser.add_argument('--val_label_dir', type=str, default='./data/fivek/val/label/')
    parser.add_argument('--test_label_dir', type=str, default='./data/fivek/test/label/')
    parser.add_argument('--model_save_path', type=str, default='models')
    parser.add_argument('--sample_path', type=str, default='samples')
    parser.add_argument('--log_path', type=str, default='logs')
    parser.add_argument('--val_result_path', type=str, default='validation')
    parser.add_argument('--test_result_path', type=str, default='test')

    # step size
    parser.add_argument('--log_step', type=int, default=100)
    parser.add_argument('--info_step', type=int, default=100)
    parser.add_argument('--sample_step', type=int, default=100)
    parser.add_argument('--model_save_epoch', type=int, default=1) 

    # Misc    
    parser.add_argument('--parallel', type=str2bool, default=False, help='use multi-GPU for training')
    parser.add_argument('--gpu_ids', default=[0, 1, 2, 3])
    parser.add_argument('--use_tensorboard', type=str, default=False)
    parser.add_argument('--is_print_network', type=str2bool, default=True)
    parser.add_argument('--is_test_nima', type=str2bool, default=True)
    parser.add_argument('--is_test_psnr_ssim', type=str2bool, default=False)
    parser.add_argument('--psnr_save_path', type=str, default='./results/psnr_val_results/')
    parser.add_argument('--ssim_save_path', type=str, default='./results/ssim_val_results')
    parser.add_argument('--nima_result_save_path', type=str, default='./results/nima_val_results/')

    return parser.parse_args()