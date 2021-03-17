import argparse
import os
import numpy as np
from tqdm import tqdm

# import utils.preddataset as preddataset

from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from modeling.pspnet import *
from modeling.Segnet import *
from modeling.HRNet import *
from modeling.HRNet_OCR import *
from modeling.HRNet_OCR_SNws import *
from modeling.UNet_SNws import *
from modeling.UNet_ac import *
from modeling.UNet3p_SNws import *
from modeling.UNet3p_res_ocr_SNws import *
from modeling.UNet3p_res_edge_aspp_SNws import *
from modeling.gscnn import *
from modeling.HUNet import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from architect import Architect

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class Trainer(object):

    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        # 使用tensorboardX可视化
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        
        # Define Dataloader
#        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        kwargs = {'num_workers': 0, 'pin_memory': True}
        #self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)
        self.train_loader , self.val_loader, self.nclass = make_data_loader(args, **kwargs)

        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset + '_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        # Define Criterion        
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)

        if args.model_name == 'unet':
            #model = UNet_ac(args.n_channels, args.n_filters, args.n_class).cuda()
            model = UNet_SNws(args.n_channels, args.n_filters, args.n_class, args.using_movavg, args.using_bn).cuda()
            model = UNet_bn(args.n_channels, args.n_filters, args.n_class).cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            #optimizer = torch.optim.AdamW(model.parameters(), lr=args.arch_lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
            #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
        # elif args.model_name == 'hunet':
        #     model = HUNet(args.n_channels, args.n_filters, args.n_class, args.using_movavg, args.using_bn).cuda()
        #     optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.model_name == 'unet3+':
            model = UNet3p_SNws(args.n_channels, args.n_filters, args.n_class, args.using_movavg, args.using_bn).cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.model_name == 'unet3+_aspp':
            #model = UNet3p_aspp(args.n_channels, args.n_filters, args.n_class, args.using_movavg, args.using_bn).cuda()
            #model = UNet3p_aspp_SNws(args.n_channels, args.n_filters, args.n_class, args.using_movavg, args.using_bn).cuda()
            #model = UNet3p_res_aspp_SNws(args.n_channels, args.n_filters, args.n_class, args.using_movavg, args.using_bn).cuda()
            model = UNet3p_res_edge_aspp_SNws(args.n_channels, args.n_filters, args.n_class, args.using_movavg, args.using_bn).cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.model_name == 'unet3+_ocr':
            model = UNet3p_res_ocr_SNws(args.n_channels, args.n_filters, args.n_class, args.using_movavg, args.using_bn).cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # elif args.model_name == 'unet3+_resnest_aspp':
        #     model = UNet3p_resnest_aspp(args.n_channels, args.n_filters, args.n_class, args.using_movavg, args.using_bn).cuda()
        #     optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.model_name == 'gscnn':
            model = GSCNN(args.n_channels, args.n_filters, args.n_class, args.using_movavg, args.using_bn).cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.model_name == 'pspnet':
            model = PSPNet(args.n_channels, args.n_filters, args.n_class).cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.model_name == 'segnet':
            model = Segnet(args.n_channels, args.n_filters, args.n_class).cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.model_name == 'hrnet':
            MODEL = {'ALIGN_CORNERS': True,
                     'EXTRA': {'FINAL_CONV_KERNEL': 1,      # EXTRA 具体定义了模型的结果，包括 4 个 STAGE，各自的参数
                               'STAGE1': {'NUM_MODULES': 1,    # HighResolutionModule 重复次数
                                          'NUM_BRANCHES': 1,   # 分支数
                                          'BLOCK': 'BOTTLENECK',
                                          'NUM_BLOCKS': 4,
                                          'NUM_CHANNELS': 64,
                                          'FUSE_METHOD': 'SUM'
                                          },
                               'STAGE2': {'NUM_MODULES': 1,
                                          'NUM_BRANCHES': 2,
                                          'BLOCK': 'BASIC',
                                          'NUM_BLOCKS': [4, 4],
                                          'NUM_CHANNELS': [48, 96],
                                          'FUSE_METHOD': 'SUM'
                                          },
                               'STAGE3': {'NUM_MODULES': 4,
                                          'NUM_BRANCHES': 3,
                                          'BLOCK': 'BASIC',
                                          'NUM_BLOCKS': [4, 4, 4],
                                          'NUM_CHANNELS': [48, 96, 192],
                                          'FUSE_METHOD': 'SUM'
                                          },
                               'STAGE4': {'NUM_MODULES': 3,
                                          'NUM_BRANCHES': 4,
                                          'BLOCK': 'BASIC',
                                          'NUM_BLOCKS': [4, 4, 4, 4],
                                          'NUM_CHANNELS': [48, 96, 192, 384],
                                          'FUSE_METHOD': 'SUM'
                                          }
                               }
                     }
            model = HighResolutionNet(args.n_channels, args.n_filters, args.n_class, MODEL).cuda()
            # model.init_weights()
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.model_name == 'hrnet+ocr':
            MODEL = {'ALIGN_CORNERS': True,
                     'EXTRA': {'FINAL_CONV_KERNEL': 1,      # EXTRA 具体定义了模型的结果，包括 4 个 STAGE，各自的参数
                               'STAGE1': {'NUM_MODULES': 1,    # HighResolutionModule 重复次数
                                          'NUM_BRANCHES': 1,   # 分支数
                                          'BLOCK': 'BOTTLENECK',
                                          'NUM_BLOCKS': 4,
                                          'NUM_CHANNELS': 64,
                                          'FUSE_METHOD': 'SUM'
                                          },
                               'STAGE2': {'NUM_MODULES': 1,
                                          'NUM_BRANCHES': 2,
                                          'BLOCK': 'BASIC',
                                          'NUM_BLOCKS': [4, 4],
                                          'NUM_CHANNELS': [48, 96],
                                          'FUSE_METHOD': 'SUM'
                                          },
                               'STAGE3': {'NUM_MODULES': 4,
                                          'NUM_BRANCHES': 3,
                                          'BLOCK': 'BASIC',
                                          'NUM_BLOCKS': [4, 4, 4],
                                          'NUM_CHANNELS': [48, 96, 192],
                                          'FUSE_METHOD': 'SUM'
                                          },
                               'STAGE4': {'NUM_MODULES': 3,
                                          'NUM_BRANCHES': 4,
                                          'BLOCK': 'BASIC',
                                          'NUM_BLOCKS': [4, 4, 4, 4],
                                          'NUM_CHANNELS': [48, 96, 192, 384],
                                          'FUSE_METHOD': 'SUM'
                                          }
                               }
                     }
            # model = HighResolutionNet_OCR(args.n_channels, args.n_filters, args.n_class, MODEL).cuda()
            model = HighResolutionNet_OCR_SNws(args.n_channels, args.n_filters, args.n_class, MODEL).cuda()
            # model.init_weights()
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.model_name == 'deeplabv3+':
        # Define network
            model = DeepLab(num_classes=self.nclass,
                            backbone=args.backbone,
                            output_stride=args.out_stride,
                            sync_bn=args.sync_bn,
                            freeze_bn=args.freeze_bn)
            backbone = model.backbone
            
#             backbone.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            print('change the input channels', backbone.conv1) 
            
            train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                            {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]
              
            #optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
            #                            weight_decay=args.weight_decay, nesterov=args.nesterov)
            optimizer = torch.optim.AdamW(train_params, weight_decay=args.weight_decay)
            #optimizer = torch.optim.AdamW(train_params, lr=args.arch_lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
        #elif args.model_name == 'autodeeplab':
            #model = AutoDeeplab(args.n_class, 12, self.criterion, crop_size=args.crop_size)
            #optimizer = torch.optim.AdamW(model.weight_parameters(), lr=args.lr, weight_decay=args.weight_decay)
            #optimizer = torch.optim.SGD(model.weight_parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        self.model, self.optimizer = model, optimizer
        
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        #self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
        #                                    args.epochs, len(self.train_loader))
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(self.train_loader))
        
        # Using cuda
        if args.cuda:
            print(self.args.gpu_ids)
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            # print(image.shape)
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)       
            
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            # Show 10 * 3 inference results each epoch
            if i % (num_img_tr // 10) == 0:
                global_step = i + num_img_tr * epoch
#                 self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)  # 保存标量值
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % (train_loss / len(tbar)))
        # print('Loss: %.3f' % (train_loss / i))

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()  # 创建全为0的混淆矩阵
        tbar = tqdm(self.val_loader, desc='\r')  # 回车符
        val_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
#            image, target = sample[0], sample[1]
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            val_loss += loss.item()
            tbar.set_description('Val loss: %.3f' % (val_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)  # 按行
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()  # 频权交并比
        self.writer.add_scalar('val/total_loss_epoch', val_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % (val_loss / len(tbar)))
        
        new_pred = FWIoU    # mIoU

        # log
        logfile = os.path.join('/home/wzj/mine_cloud_14/','log.txt')
        log_file = open(logfile, 'a')
        if epoch == 0:
            log_file.seek(0)
            log_file.truncate()
            log_file.write(self.args.model_name + '\n')
        log_file.write('Epoch: %d, ' % (epoch + 1))
        if new_pred < self.best_pred:
            log_file.write('Acc: {}, Acc_class: {}, mIoU: {}, fwIoU: {}, best_fwIoU: {}, '.format(Acc, Acc_class, mIoU, FWIoU, self.best_pred))
        else:
            log_file.write('Acc: {}, Acc_class: {}, mIoU: {}, fwIoU: {}, best_fwIoU: {}, '.format(Acc, Acc_class, mIoU, FWIoU, new_pred))
        log_file.write('Loss: %.3f\n' % (val_loss / len(tbar)))
        if epoch == 199:   # 499
            log_file.close()     

        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")  # 创建解析器
    parser.add_argument('--backbone', type=str, default='xception',
                        choices=['resnet', 'xception', 'drn', 'mobilenet', 'resnest'],  # drn：深度残差网络
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='cloud',
                        choices=['pascal', 'coco', 'cityscapes', 'cloud'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--use-sbd', action='store_true', default=True,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=0,  # default=4
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=512,  # default=513
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=512,  # default=513
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,  # 同步
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,  # 冻结
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal', 'ce+focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: auto)')  # metavar参数:用来控制部分命令行参数的显示
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch_size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,  # False
                        help='whether to use balanced weights (default: False)')
                        # 'balanced'计算出来的结果很均衡，使得惩罚项和样本量对应，惩罚项用的样本数的倒数
    # optimizer params
    parser.add_argument('--lr', type=float, default=0.002, metavar='LR',
                        help='learning rate (default: auto)')  # 0.005
    parser.add_argument('--arch_lr', type=float, default=1e-3,
                        help='learning rate for alpha and beta in architect searching process') #3e-3
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')   # 权重衰减
    parser.add_argument('--arch_weight_decay', type=float, default=5e-4,
                        metavar='M', help='w-decay(default:5e-4)')   # 1e-3
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')  # Nesterov牛顿动量法是Momentum的变种
                        # 与Momentum唯一区别就是，计算梯度的不同，Nesterov先用当前的速度v更新一遍参数，再用更新的临时参数计算梯度
                        # 相当于添加了矫正因子的Momentum
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0,1',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')  # comma-separated:逗号分割
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')  # 恢复文件
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
                        # action='store_true':只要运行时该变量有传参就将该变量设为True
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')
    
    parser.add_argument('--model_name', type=str, default='hrnet+ocr', choices=['deeplabv3+', 'unet', 'hunet', 'unet3+', 'unet3+_aspp', 'unet3+_ocr', 'unet3+_resnest_aspp','gscnn', 'pspnet', 'hrnet', 'segnet', 'hrnet+ocr'])
    parser.add_argument('--n_channels', type=int, default=14)
    parser.add_argument('--n_filters', type=int, default=64)
    parser.add_argument('--n_class', type=int, default=10)
    parser.add_argument('--using_movavg', type=int, default=1)
    parser.add_argument('--using_bn', type=int, default=1)

    args = parser.parse_args()
    # parser.parse_args():把parser中设置的所有"add_argument"返回到args子类实例当中，那么parser中增加的属性内容都会在args实例中，使用即可
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')  # 用raise语句来引发一个异常

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 200,
            'pascal': 50,
            'cloud':200
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        #args.batch_size = 8 * len(args.gpu_ids)
        args.batch_size = 2

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.007,
            'cloud':0.005
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size

    if args.checkname is None:
        if args.model_name == 'deeplabv3+':
            args.checkname = 'deeplab-' + str(args.backbone)  # + '2020-10-16'
        elif args.model_name == 'hunet':
            args.checkname = 'hunet'
        elif args.model_name == 'unet':
            args.checkname = 'unet'
        elif args.model_name == 'unet3+':
            args.checkname = 'unet3+'
        elif args.model_name == 'unet3+_aspp':
            args.checkname = 'unet3+_aspp'
        elif args.model_name == 'unet3+_resnest_aspp':
            args.checkname = 'unet3+_resnest_aspp'
        elif args.model_name == 'gscnn':
            args.checkname = 'gscnn'
        elif args.model_name == 'pspnet':
            args.checkname = 'pspnet'
        elif args.model_name == 'hrnet':
            args.checkname = 'hrnet'
        elif args.model_name == 'segnet':
            args.checkname = 'segnet'
        elif args.model_name == 'hrnet+ocr':
            args.checkname = 'hrnet+ocr'
        elif args.model_name == 'unet3+_ocr':
            args.checkname = 'unet3+_ocr'

    print(args)
    torch.manual_seed(args.seed)  # 设置 (CPU/GPU) 生成随机数的种子，并返回一个torch.Generator对象
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)

    trainer.writer.close()


if __name__ == "__main__":
    main()
