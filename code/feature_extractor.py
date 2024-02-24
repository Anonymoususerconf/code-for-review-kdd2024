# -*- coding: utf-8 -*-

import os
import json
import PIL.Image as pil
import random
import time
import datetime
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import argparse
from utils import * 
from dataset_feature import *
from models import FeatureExtractor
from configuration import config


class Trainer(object):
    def __init__(self, config):
        super(Trainer, self).__init__()
        self.config = config
        file_path = os.path.abspath(__file__)
        self.bert_chinese_path = os.path.dirname(file_path) + '/../bert-base-chinese/'
        self.region_emb_save_path = os.path.dirname(file_path) + '/../region_emb/{c}.pth'.format(c=config['city'])
        self.pretrain_state_path = os.path.dirname(file_path) + '/../checkpoint/'
        
    
    def get_dataloader(self):
        dataset = Dataset_RegionFeature(config=self.config)
        dataloader = DataLoader(dataset=dataset, batch_size=self.config['extract_batch_size'], shuffle=False, num_workers=2)
        print(len(dataset), len(dataloader))
        return dataloader
    

    def load_model(self):
        model = FeatureExtractor(config=self.config).to('cuda')
        checkpoint_path = self.pretrain_state_path + 'checkpoint-{}.pth'.format(self.config['checkpoint'])
        print("checkpoint_path:\n" + checkpoint_path)
        pretrain_state = torch.load(checkpoint_path)
        pretrain_model = pretrain_state['model']
        model.load_state_dict(pretrain_model, strict=False)
        return model


    def train(self):
        device = torch.device('cuda')
        dataloader = self.get_dataloader()
        model = self.load_model()

        self.feature_extraction(
            model=model, 
            dataloader=dataloader, 
            device=device,
        )


    @torch.no_grad()
    def feature_extraction(self, model, dataloader, device):
        region_emb_list = []
        model.eval()
        for step, data in enumerate(dataloader):

            poi_name_token_ids = data['poi_name_token_ids'].to(device, non_blocking=True)
            attn_mask_poi = data['attn_mask_poi'].to(device, non_blocking=True)
            word_level_pos_ids = data['word_level_pos_ids'].to(device, non_blocking=True)
            poi_level_pos_ids = data['poi_level_pos_ids'].to(device, non_blocking=True)
            grid_level_pos_ids = data['grid_level_pos_ids'].to(device, non_blocking=True)
            poi_cate_ids = data['poi_cate_ids'].to(device, non_blocking=True)

            img = data['img'].to(device, non_blocking=True)

            poi_data = {
                'poi_name_token_ids': poi_name_token_ids,
                'attn_mask_poi': attn_mask_poi,
                'word_level_pos_ids': word_level_pos_ids,
                'poi_level_pos_ids': poi_level_pos_ids,
                'grid_level_pos_ids': grid_level_pos_ids,
                'poi_cate_ids': poi_cate_ids,
            }

            img_data = {
                'img': img,
            }

            region_emb = model(poi_data=poi_data, img_data=img_data)
            region_emb_list.append(region_emb.cpu())

            print("extract {} steps".format(step))
            
        
        region_emb_list = torch.cat(region_emb_list, dim=0)
        print(region_emb_list.shape)
        self.save_region_emb(region_emb_list=region_emb_list)


    def Train(self):
        seed_setup(self.config['seed'])
        self.log = str(self.config) + '\n------------------- start training ----------------------\n'
        self.train()


    def save_region_emb(self, region_emb_list):
        region_emb_save_path = self.region_emb_save_path
        if os.path.exists(region_emb_save_path):
            os.system('rm ' + region_emb_save_path)
        
        torch.save(region_emb_list, region_emb_save_path)


if __name__ == '__main__':
    pass
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default=None)
    parser.add_argument('--checkpoint', type=int, default=None)
    parser.add_argument('--extract_batch_size', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    for k, v in vars(args).items():
        config[k] = v
    
    for k, v in config.items():
        print('{}: '.format(k), v, type(v))
    
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    Trainer_ = Trainer(config=config)
    Trainer_.Train()