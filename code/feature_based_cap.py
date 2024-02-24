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
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import argparse
from utils import * 
from models import FeatureBasedCommercialActivenessPrediction
from configuration import config



class FeatureBased_Dataset_CAP(Dataset):
    def __init__(self, config=None, dataset_type=None):
        super().__init__()
        self.config = config
        file_path = os.path.abspath(__file__)
        self.root_path = os.path.dirname(file_path) + '/../'
        self.dataset_type = dataset_type
        self.id_of_region_path = self.root_path + 'data/cap_{c}_id_of_region_{dt}'.format(c=config['city'], dt=dataset_type)
        self.label_path = self.root_path + 'data/cap_{c}_label_{dt}'.format(c=config['city'], dt=dataset_type)
        self.region_emb_path = self.root_path + 'region_emb/{c}.pth'.format(c=config['city'])

        self.rehash_id_of_region = []
        with open(self.id_of_region_path, 'r') as f:
            for line in f:
                line = eval(line.strip('\n'))
                rehash_id = int(line[0])
                self.rehash_id_of_region.append(rehash_id)

        label_list = []
        with open(self.label_path, 'r') as f:
            for line in f:
                line = eval(line.strip('\n'))
                label_list.append(line)

        self.label_list = [float(label) for label in label_list]

        region_emb_total = torch.load(self.region_emb_path)
        self.region_emb = region_emb_total[self.rehash_id_of_region]


    def __getitem__(self, index):
        label = self.label_list[index]
        region_emb = self.region_emb[index]
        return {'label': label, 'region_emb': region_emb}


    def __len__(self):
        return len(self.rehash_id_of_region)




class Trainer(object):
    def __init__(self, config):
        super(Trainer, self).__init__()
        self.config = config
        file_path = os.path.abspath(__file__)
        self.log_path = os.path.dirname(file_path) + '/../log/feature_based_cap_log_{c}.txt'.format(c=config['city'])

    
    def get_dataloader(self):
        train_dataset = FeatureBased_Dataset_CAP(config=self.config, dataset_type='train')
        val_dataset = FeatureBased_Dataset_CAP(config=self.config, dataset_type='val')
        test_dataset = FeatureBased_Dataset_CAP(config=self.config, dataset_type='test')
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.config['fbatch_size'], shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.config['fbatch_size'], shuffle=False)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.config['fbatch_size'], shuffle=False)
        print(len(train_dataset), len(train_loader), len(val_dataset), len(val_loader), len(test_dataset), len(test_loader))
        return train_loader, val_loader, test_loader
    

    def load_model(self):
        model = FeatureBasedCommercialActivenessPrediction(config=self.config).to('cuda')
        return model
     

    def train(self):
        device = torch.device('cuda')
        train_loader, val_loader, test_loader = self.get_dataloader()
        model = self.load_model()

        param_groups = [{'params':model.parameters(), 'lr':self.config['flr'], 'weight_decay': self.config['fdecay']}]

        optimizer = optim.AdamW(param_groups, lr=self.config['flr'])
        criterion = nn.MSELoss()
        best_rmse_val, best_epoch = 999999, 0

        self.start_time = time.time()
        for epoch in range(self.config['epoch_num']):
            self.train_one_epoch(model=model, 
                                 train_loader=train_loader, 
                                 optimizer=optimizer, 
                                 epoch=epoch, 
                                 criterion=criterion,
                                 device=device)
            
            rmse_val, mae_val, R2_val = self.evaluate(model=model, 
                                        eval_loader=val_loader, 
                                        epoch=epoch, 
                                        criterion=criterion,
                                        device=device)

            rmse_test, mae_test, R2_test = self.evaluate(model=model, 
                                        eval_loader=test_loader, 
                                        epoch=epoch, 
                                        criterion=criterion,
                                        device=device)

            if rmse_val < best_rmse_val:
                best_epoch = epoch
                best_rmse_val, best_mae_val, best_R2_val = rmse_val, mae_val, R2_val
                best_rmse_test, best_mae_test, best_R2_test = rmse_test, mae_test, R2_test
                star = '*** '
            else: star = ''
            
            log_new = star + "Epoch: %3d | Val RMSE: %.4f | Val MAE: %.4f | Val R2: %.4f | Test RMSE: %.4f | Test MAE: %.4f | Test R2: %.4f \n" \
                    % (epoch, rmse_val, mae_val, R2_val, rmse_test, mae_test, R2_test)
            self.log = self.log + log_new + '\n'
            print(log_new)

        log_new = "Best Epoch: %3d | Best Val RMSE: %.4f | Best Val MAE: %.4f | Best Val R2: %.4f | Best Test RMSE: %.4f | Best Test MAE: %.4f | best test R2: %.4f " \
                    % (best_epoch, best_rmse_val, best_mae_val, best_R2_val, best_rmse_test, best_mae_test, best_R2_test)
        self.log = self.log + log_new + '\n'
        print(log_new)


    def train_one_epoch(self, model, train_loader, optimizer, epoch, criterion, device):
        model.train()
        optimizer.zero_grad()

        for step, data in enumerate(train_loader):
            label = data['label'].float().to(device, non_blocking=True)

            region_emb = data['region_emb'].to(device, non_blocking=True)
            
            if step % self.config['accum_iter'] == 0:
                adjust_learning_rate(
                    optimizer=optimizer, 
                    epoch=step / len(train_loader) + epoch, 
                    warmup_epochs=self.config['warmup_epochs'],
                    epoch_num=self.config['epoch_num'],
                    peak_lr=self.config['flr'],
                    min_lr=self.config['min_lr'])

            pred = model(region_emb=region_emb)
            pred = pred.squeeze(-1)
            loss = criterion(pred, label)
        
            loss_value = loss.item()
            
            loss = loss / self.config['accum_iter']
            loss.backward()

            if (step + 1) % self.config['accum_iter'] == 0:
                optimizer.step()
                optimizer.zero_grad()

            torch.cuda.synchronize()
            total_time = time.time() - self.start_time
            total_time = str(datetime.timedelta(seconds=int(total_time)))

            if step % config['logging_step'] == 0:
                log_new = "Epoch: %3d | Step: %4d | Train Loss: %7.4f | Time: %s" % (epoch, step, loss_value, total_time)
                self.log = self.log + log_new + '\n'
                print(log_new)


    @torch.no_grad()
    def evaluate(self, model, eval_loader, epoch, criterion, device):
        eval_pred_list, eval_label_list = [], []
        model.eval()
        for _, data in enumerate(eval_loader):
            eval_label_list += data['label'].tolist()

            region_emb = data['region_emb'].to(device, non_blocking=True)

            pred = model(region_emb=region_emb)
            pred = pred.squeeze(-1)
            eval_pred_list += pred.to('cpu').tolist()
        
        rmse_eval = np.sqrt(mean_squared_error(eval_label_list, eval_pred_list))
        mae_eval = mean_absolute_error(eval_label_list, eval_pred_list)
        R2_eval = r2_score(eval_label_list, eval_pred_list)
        return rmse_eval, mae_eval, R2_eval
            
    def Train(self):
        seed_setup(self.config['seed'])
        self.log = str(self.config) + '\n------------------- start training ----------------------\n'
        self.train()
        self.write_log()

    def write_log(self):
        log_output_path = self.log_path
        if os.path.exists(log_output_path):
            os.system('rm ' + log_output_path)
        with open(log_output_path, 'w') as f:
            f.write(self.log)


if __name__ == '__main__':
    pass
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default=None)
    parser.add_argument('--checkpoint', type=int, default=None)
    parser.add_argument('--agg', type=str, default=None)
    parser.add_argument('--fdrop', type=float, default=None)
    parser.add_argument('--fbatch_size', type=int, default=None)
    parser.add_argument('--epoch_num', type=int, default=None)
    parser.add_argument('--warmup_epochs', type=int, default=None)
    parser.add_argument('--flr', type=float, default=None)
    parser.add_argument('--fdecay', type=float, default=None)
    parser.add_argument('--min_lr', type=float, default=None)
    parser.add_argument('--accum_iter', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--logging_step', type=int, default=10)
    args = parser.parse_args()

    for k, v in vars(args).items():
        config[k] = v

    for k, v in config.items():
        print('{}: '.format(k), v, type(v))
    
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    Trainer_ = Trainer(config=config)
    Trainer_.Train()