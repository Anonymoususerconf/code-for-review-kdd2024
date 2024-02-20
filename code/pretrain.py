# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import datetime
import PIL.Image as pil
import random
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
from utils import * 
from dataset import Pretrain_Dataset
from models import ModelForPretraining
from configuration import config
import utils_dist
import torch.distributed as dist


class Trainer(object):
    def __init__(self, args, config):
        super(Trainer, self).__init__()
        self.args = args
        self.config = config
        file_path = os.path.abspath(__file__)
        self.log_path = os.path.dirname(file_path) + '/../log/pretrain_log.txt'
        self.save_path = os.path.dirname(file_path) + '/../checkpoint/'
   

    def get_dataloader(self):
        train_dataset = Pretrain_Dataset(config=self.config)

        world_size = utils_dist.get_world_size()
        global_rank = utils_dist.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=global_rank, shuffle=True
        )

        train_loader = DataLoader(
            dataset=train_dataset, 
            sampler=sampler_train,
            batch_size=self.config['batch_size'], 
            # num_workers=2,
            pin_memory=True,
            drop_last=True
        )
        print(len(train_dataset), len(train_loader))
        return train_loader


    
    def train(self):
        if self.args.distributed:
            device = torch.device('cuda', self.args.local_rank)
        else:
            device = torch.device('cuda')

        train_loader = self.get_dataloader()

        model = ModelForPretraining(config=self.config).to(device)
        model_without_ddp = model

        if self.args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.args.local_rank], find_unused_parameters=True)
            model_without_ddp = model.module

        param_list = [{'params':model.parameters(), 'lr': self.config['lr'], 'weight_decay': self.config['decay']}]
        optimizer = optim.AdamW(param_list)

        self.args.start_epoch = 0

        self.start_time = time.time()
        for epoch in range(self.args.start_epoch, self.config['epoch_num']):
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            self.train_one_epoch(
                model=model, 
                train_loader=train_loader, 
                optimizer=optimizer, 
                device=device,
                epoch=epoch, 
            )
            
            self.save_model(
                model_without_ddp=model_without_ddp, 
                epoch=epoch, 
                optimizer=optimizer, 
            )
                

    def train_one_epoch(self, model, train_loader, optimizer, device, epoch):
        
        model.train()
        optimizer.zero_grad()
        
        for step, data in enumerate(train_loader):

            poi_name_token_ids_masked = data['poi_name_token_ids_masked'].to(device, non_blocking=True)
            attn_mask_poi = data['attn_mask_poi'].to(device, non_blocking=True)
            word_level_pos_ids = data['word_level_pos_ids'].to(device, non_blocking=True)
            poi_level_pos_ids = data['poi_level_pos_ids'].to(device, non_blocking=True)
            grid_level_pos_ids = data['grid_level_pos_ids'].to(device, non_blocking=True)
            poi_cate_ids = data['poi_cate_ids'].to(device, non_blocking=True)
            poi_masked_label = data['poi_masked_label'].to(device, non_blocking=True)

            llm_poi_feat = data['llm_poi_feat'].to(device, non_blocking=True)
                            
            img = data['img'].to(device, non_blocking=True)
            img_mask = data['img_mask'].to(device, non_blocking=True)
            img_masked_label = data['img_masked_label'].to(device, non_blocking=True)

            vfm_img_feat = data['vfm_img_feat'].to(device, non_blocking=True)
            

            align_label = data['align_label'].to(device, non_blocking=True)
            vlfm_img_feat = data['vlfm_img_feat'].to(device, non_blocking=True)
                        

            poi_data = {
                'poi_name_token_ids_masked': poi_name_token_ids_masked,
                'attn_mask_poi': attn_mask_poi,
                'word_level_pos_ids': word_level_pos_ids,
                'poi_level_pos_ids': poi_level_pos_ids,
                'grid_level_pos_ids': grid_level_pos_ids,
                'poi_cate_ids': poi_cate_ids,
            }

            img_data = {
                'img': img,
                'img_mask': img_mask,
            }
            

            if step % self.config['accum_iter'] == 0:
                adjust_learning_rate(
                    optimizer=optimizer,
                    epoch=float(step) / len(train_loader) + epoch,
                    warmup_epochs=self.config['warmup_epochs'],
                    epoch_num=self.config['epoch_num'],
                    peak_lr=self.config['lr'],
                    min_lr=self.config['min_lr'],
                )

            return_loss_dict = model(
                poi_data=poi_data,
                img_data=img_data,
                poi_masked_label=poi_masked_label,
                img_masked_label=img_masked_label,
                align_label=align_label,
                llm_poi_feat=llm_poi_feat,
                vfm_img_feat=vfm_img_feat,
                vlfm_img_feat=vlfm_img_feat,
            )

            poi_mgdm_loss = return_loss_dict['poi_mgdm_loss']
            img_mgdm_loss = return_loss_dict['img_mgdm_loss']
            cmsa_loss = return_loss_dict['cmsa_loss']
            dlfm_loss = return_loss_dict['dlfm_loss']
            dvfm_loss = return_loss_dict['dvfm_loss']
            dvlfm_loss = return_loss_dict['dvlfm_loss']
            
            loss = poi_mgdm_loss + img_mgdm_loss + cmsa_loss + dlfm_loss + dvfm_loss + dvlfm_loss
            loss_value = loss.item()

            poi_mgdm_loss_value = poi_mgdm_loss if isinstance(poi_mgdm_loss, float) else poi_mgdm_loss.item()
            img_mdgm_loss_value = img_mgdm_loss if isinstance(img_mgdm_loss, float) else img_mgdm_loss.item()
            cmsa_loss_value = cmsa_loss if isinstance(cmsa_loss, float) else cmsa_loss.item()
            dlfm_loss_value = dlfm_loss if isinstance(dlfm_loss, float) else dlfm_loss.item()
            dvfm_loss_value = dvfm_loss if isinstance(dvfm_loss, float) else dvfm_loss.item()
            dvlfm_loss_value = dvlfm_loss if isinstance(dvlfm_loss, float) else dvlfm_loss.item()

            loss = loss / self.config['accum_iter']
            loss.backward()

            if (step + 1) % self.config['accum_iter'] == 0:
                optimizer.step()
                optimizer.zero_grad()

            torch.cuda.synchronize()
            total_time = time.time() - self.start_time
            total_time = str(datetime.timedelta(seconds=int(total_time)))
            
            if step % config['logging_step'] == 0:
                log_new = "Epoch:%3d | Step:%5d | Loss:%7.4f | MGDM_P:%7.4f | MGDM_S:%7.4f | CMSA:%7.4f | DLFM:%7.4f | DVFM:%7.4f | DVLFM:%7.4f | Time:%s" % (
                    epoch, step, loss_value, 
                    poi_mgdm_loss_value, img_mdgm_loss_value, cmsa_loss_value,
                    dlfm_loss_value, dvfm_loss_value, dvlfm_loss_value, 
                    total_time
                )
                
                self.log = self.log + log_new + '\n'
                print(log_new)

            if self.args.distributed:
                dist.barrier()

        self.write_log()
 

    def save_model(self, model_without_ddp, epoch, optimizer):
        if (epoch+1) % self.config['save_epochs'] == 0 or epoch == (self.config['epoch_num']-1):
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'config': self.config
            }
            save_path = self.save_path + 'checkpoint-{}.pth'.format(epoch)
            assert not os.path.exists(save_path), "model dir exists."
            utils_dist.save_on_master(to_save, save_path)


    def Train(self):

        seed_setup(self.config['seed'])
        self.log = ''
        for k, v in self.config.items():
            log_new = '{}: {} {}'.format(k, v, type(v))
            print(log_new)
            self.log = self.log + log_new + '\n'
        self.log = self.log + '\n------------------- start pretraining ----------------------\n'
        self.train()
        self.write_log()
    

    def write_log(self):

        if utils_dist.is_main_process():
            log_output_path = self.log_path
            if os.path.exists(log_output_path):
                os.system('rm ' + log_output_path)
            with open(log_output_path, 'w') as f:
                f.write(self.log)
                
 

if __name__ == '__main__':
    pass
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epoch_num', type=int, default=None)
    parser.add_argument('--warmup_epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--min_lr', type=float, default=None)
    parser.add_argument('--decay', type=float, default=None)
    parser.add_argument('--accum_iter', type=int, default=None)
    parser.add_argument('--save_epochs', type=int, default=None)
    parser.add_argument('--logging_step', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--local_rank', default=-1, type=int)
    args = parser.parse_args()

    utils_dist.init_distributed_mode(args)

    for k, v in vars(args).items():
        config[k] = v
    
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    Trainer_ = Trainer(args=args, config=config)
    Trainer_.Train()

    
