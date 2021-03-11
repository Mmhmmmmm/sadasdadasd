import argparse
import json
import os
from os.path import join
from data import CelebA
import torch
import torch.utils.data as data
import torchvision.utils as vutils

from attgan import AttGAN
from data import check_attribute_conflict
import cv2

def parse(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', dest='experiment_name', type=str, default='06-36AM on March 09, 2021')
    parser.add_argument('--gpu', dest='gpu', type=bool, default=True)
    parser.add_argument('--use_model', dest='use_model', type=str, default='generator')
    parser.add_argument('--data_path', dest='data_path', type=str, default='data\CelebA\Img\img_align_celeba')
    parser.add_argument('--attr_path', dest='attr_path', type=str, default='data\CelebA\Anno\list_attr_celeba.txt')
    return parser.parse_args(args)

args_ = parse()
with open(join('output', args_.experiment_name, 'setting.txt'), 'r') as f:
    args = json.load(f, object_hook=lambda d: argparse.Namespace(**d))

args.gpu = args_.gpu
args.n_attrs = len(args.attrs)
args.betas = (args.beta1, args.beta2)
args.data_path = args_.data_path
args.attr_path = args_.attr_path

test_dataset = CelebA(args.data_path, args.attr_path, args.img_size, 'mytest', args.attrs)
test_dataloader = data.DataLoader(
    test_dataset, batch_size=1, num_workers=args.num_workers,
    shuffle=False, drop_last=False
)

print('Testing images:', len(test_dataset))

output_path = join('output', args.experiment_name, 'attention_testing')
os.makedirs(output_path, exist_ok=True)

attgan = AttGAN(args)
attgan.load(r'weights_unzip.17.pth')
attgan.eval()

for idx, (img_real, att_org) in enumerate(test_dataloader):
    img_real = img_real.cuda() if args.gpu else img_real
    att_org = att_org.cuda() if args.gpu else att_org
    att_org = att_org.type(torch.float)

    att_list = [att_org]
    for i in range(args.n_attrs):
        tmp = att_org.clone()
        tmp[:, i] = 1 - tmp[:, i]
        tmp = check_attribute_conflict(tmp, args.attrs[i], args.attrs)
        att_list.append(tmp)

    if args_.use_model == 'generator':
        with torch.no_grad():
            samples = [img_real]
            for i, att_tar in enumerate(att_list):
                if i > 0:
                    att_diff = att_tar - att_org
                    samples.append(attgan.G(img_real, att_diff))
            samples = torch.cat(samples, dim=3)
            out_file = '{:06d}.jpg'.format(idx)
            vutils.save_image(
                samples, join(output_path, out_file),
                nrow=1, normalize=True, range=(-1., 1.)
            )
            print('{:s} done!'.format(out_file))
    elif args_.use_model == 'discriminator':
        with torch.no_grad():
            samples = []
            abn_att_real, cabn_att_real = attgan.D(img_real, 'abn')
            # num,  w, h = abn_att_real[0].size()
            print(abn_att_real.shape)
            
            #print(att_org)
            #print(len(att_list))
            print(abn_att_real,cabn_att_real)
            '''
            for i in range(13):
               # print(i)
                abn_att = abn_att_real[0,i,:,:]
                cabn_att = cabn_att_real[0,i,:,:]
                # abn_att = cv2.resize(abn_att,(num, w, h))
                print(1-att_org[0,i],cabn_att.min(),cabn_att.max(),cabn_att.mean(),cabn_att.std())
                samples.append(cabn_att)
            samples = torch.cat(samples, dim=1)
            out_file = '{:06d}_c.jpg'.format(idx)
            vutils.save_image(
                samples, join(output_path, out_file),
                nrow=1, normalize=True, range=(0., 1.)
            )
            print('{:s} done!'.format(out_file))
            
            '''