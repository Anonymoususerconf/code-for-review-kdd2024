# -*- coding: utf-8 -*-

import os
import json
import PIL.Image as pil
import random
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torchvision import transforms
import numpy as np
from utils import *


class Pretrain_Dataset(Dataset):
    def __init__(self, config):
        super(Pretrain_Dataset, self).__init__()
        self.config = config
        file_path = os.path.abspath(__file__)
        self.root_path = os.path.dirname(file_path) + '/../'
        self.id_of_region_path = self.root_path + 'data/id_of_region'
        self.poi_sortby_zorder_path = self.root_path + 'data/poi_sort_by_zorder'
        self.poi_cate_vocab_path = self.root_path + 'data/poi_cate_vocab'
        self.region_coord_path = self.root_path + 'data/region_coord'
        self.bert_chinese_path = self.root_path + 'bert-base-chinese/'
        self.img_path = self.root_path + 'data/satellite_img/'
        self.visual_token_path = self.root_path + 'data/visual_token/'
        self.clip_img_feat_path = self.root_path + 'data/clip_img_feat/'
        self.llm_poi_feat_path = self.root_path + 'data/llm_poi_feat/'

        image_height, image_width = pair(config['image_size'])
        patch_height, patch_width = pair(config['patch_size'])
        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
        'Image dimensions must be divisible by the patch size.'
        self.num_patch = (image_height // patch_height) * (image_width // patch_width)

        self.max_len_token = config['max_len_token']
        self.max_len_poi = config['max_len_poi']
        self.num_grid_x = config['num_grid_x']
        self.num_grid_y = config['num_grid_y']
        self.num_grid = self.num_grid_x * self.num_grid_y

        self.id_of_region = []
        with open(self.id_of_region_path, 'r') as f:
            for line in f:
                line = line.strip('\n')
                self.id_of_region.append(line)
        
        self.dataset_len = len(self.id_of_region)

        self.poi_zorder_list = []
        with open(self.poi_sortby_zorder_path, 'r') as f:
            for line in f:
                line = eval(line.strip('\n'))
                self.poi_zorder_list.append(line[1])

        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_chinese_path)
        self.cls_token_id = self.tokenizer.cls_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.mask_token_id = self.tokenizer.mask_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        
        self.region_coord_list = []
        with open(self.region_coord_path, 'r') as f:
            for line in f:
                line = eval(line.strip('\n'))
                self.region_coord_list.append(line)
        
        self.poi_cate_vocab = {}
        with open(self.poi_cate_vocab_path, 'r') as f:
            for line in f:
                line = line.strip('\n').split('\t')
                cate, cate_id = line
                self.poi_cate_vocab[cate] = int(cate_id)
                
        mean = (0.38674183802795997, 0.41457171312761487, 0.3425218607333735)
        std = (0.17068772835394358, 0.1444371248724789, 0.16002392851072578)
        self.img_trans = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)])

            
    def tokenize_poi_data(self, index):
        poi_list = self.poi_zorder_list[index]
        region_coord = self.region_coord_list[index]

        cur_len, num_poi = 1, 1
        token_id_seq, attn_mask_seq = [self.cls_token_id], [1]
        word_level_pos_id_seq, poi_level_pos_id_seq, grid_level_pos_id_seq = [0], [0], [0]
        offset = [cur_len]
        poi_cate_id_seq = [self.poi_cate_vocab['PAD']] 
        

        for poi in poi_list:
            poi_name, _, poi_cate_id, poi_x, poi_y = poi

            poi_name = poi_name.lower()
            tokenize_output = self.tokenizer(poi_name, add_special_tokens=False)

            token_id = tokenize_output['input_ids']
            attn_mask = tokenize_output['attention_mask']

            token_id.append(self.sep_token_id) # add sep token behind each poi
            attn_mask.append(1)

            word_level_pos_id = list(range(len(token_id)))

            poi_level_pos_id = [num_poi] * len(token_id)

            grid_x = int(self.num_grid_x * (poi_x - region_coord[0]) / (region_coord[2] - region_coord[0]))
            grid_y = int(self.num_grid_y * (region_coord[1] - poi_y) / (region_coord[1] - region_coord[3]))
            grid_x, grid_y = min(grid_x, self.num_grid_x - 1), min(grid_y, self.num_grid_y - 1)
            grid_id = self.num_grid_x * grid_y + grid_x + 1
            grid_level_pos_id = [grid_id] * len(token_id)

            poi_cate_id = [poi_cate_id] * len(token_id)
            
            cur_len += len(token_id)
            if cur_len <= self.max_len_token:
                offset.append(cur_len)
                token_id_seq.extend(token_id)
                attn_mask_seq.extend(attn_mask)
                word_level_pos_id_seq.extend(word_level_pos_id)
                poi_level_pos_id_seq.extend(poi_level_pos_id)
                grid_level_pos_id_seq.extend(grid_level_pos_id)
                poi_cate_id_seq.extend(poi_cate_id)
                num_poi += 1
            
            else: 
                break

        ## padding 
        padding_len = self.max_len_token - offset[-1]
        if padding_len > 0:
            token_id_seq.extend([self.pad_token_id] * padding_len)
            attn_mask_seq.extend([0] * padding_len)
            word_level_pos_id_seq.extend([i % self.max_len_token for i in range(padding_len)])
            poi_level_pos_id_seq.extend([num_poi + int(i / self.max_len_poi) for i in range(padding_len)])
            grid_level_pos_id_seq.extend([self.num_grid + 1] * padding_len)
            poi_cate_id_seq.extend([self.poi_cate_vocab['PAD']] * padding_len)
            
        token_id_ts = torch.tensor(token_id_seq)
        attn_mask_ts = torch.tensor(attn_mask_seq)
        word_level_pos_id_ts = torch.tensor(word_level_pos_id_seq)
        poi_level_pos_id_ts = torch.tensor(poi_level_pos_id_seq)
        grid_level_pos_id_ts = torch.tensor(grid_level_pos_id_seq)
        poi_cate_id_ts = torch.tensor(poi_cate_id_seq)

        del token_id_seq, attn_mask_seq, word_level_pos_id_seq, poi_level_pos_id_seq, grid_level_pos_id_seq, poi_cate_id_seq
        
        return token_id_ts, attn_mask_ts, word_level_pos_id_ts, poi_level_pos_id_ts, grid_level_pos_id_ts, poi_cate_id_ts


    def get_masked_poi_name_token(self, poi_name_token_ids, mask_probability=0.15):
        poi_name_token_ids_new = poi_name_token_ids.clone()
        labels = poi_name_token_ids.clone()
        probability_matrix = torch.full(labels.shape, mask_probability)

        special_token_mask = self.tokenizer.get_special_tokens_mask(poi_name_token_ids_new.tolist(), already_has_special_tokens=True)
        special_token_mask = torch.tensor(special_token_mask, dtype=torch.bool)
        
        probability_matrix.masked_fill_(special_token_mask, 0.0)
        mask = torch.bernoulli(probability_matrix).bool()
        unmask = ~mask
        labels[unmask] = -100
        
        mask_replace = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & mask
        poi_name_token_ids_new[mask_replace] = self.mask_token_id

        mask_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & mask & ~mask_replace
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        poi_name_token_ids_new[mask_random] = random_words[mask_random]

        return poi_name_token_ids_new, labels, unmask
    

    def get_masked_patch_token(self, visual_token, mask_probability=0.4):
        probability_matrix = torch.full([self.num_patch], mask_probability)
        mask = torch.bernoulli(probability_matrix).bool()
        labels = torch.tensor(visual_token).flatten()
        unmask = ~mask
        labels[unmask] = -100

        return mask, labels


    def get_align_label(self, poi_name_token_ids, grid_level_pos_id, poi_unmask, img_mask):
        # exclude special tokens and masked poi name tokens
        special_token_mask = self.tokenizer.get_special_tokens_mask(poi_name_token_ids.tolist(), already_has_special_tokens=True)
        special_token_mask = torch.tensor(special_token_mask, dtype=torch.bool)
        poi_unmask = poi_unmask & ~special_token_mask

        labels = torch.full([self.config['max_len_token']], -100, dtype=torch.long)

        # the grid id of poi name tokens is from 1 to 256 in grid_level_pos_id
        # the idx of img_mask is from 0 to 255
        # align the spatial location of poi name tokens to satellite image patches
        poi_map_to_patch = grid_level_pos_id - 1
        poi_map_to_patch[poi_map_to_patch > 255] = -1
        unaligned_mask = img_mask[poi_map_to_patch] & poi_unmask
        img_unmask = ~img_mask
        aligned_mask = img_unmask[poi_map_to_patch] & poi_unmask
        labels[unaligned_mask] = 0
        labels[aligned_mask] = 1

        return labels


    def __getitem__(self, index):
        id = self.id_of_region[index]

        poi_name_token_ids, attn_mask_poi, word_level_pos_ids, poi_level_pos_ids, grid_level_pos_ids, poi_cate_ids = self.tokenize_poi_data(index)

        poi_name_token_ids_masked, poi_masked_label, poi_unmask = self.get_masked_poi_name_token(
            poi_name_token_ids=poi_name_token_ids, 
            mask_probability=self.config['mask_ratio_poi']
        )

        llm_poi_feat = torch.tensor(np.load(self.llm_poi_feat_path + '{id}.npy'.format(id=id)), dtype=torch.float32)
        
        img = pil_loader(self.img_path + '{id}.png'.format(id=id))
        img = self.img_trans(img)

        visual_token = np.load(self.visual_token_path + '{id}.npy'.format(id=id)).tolist()
        img_mask, img_masked_label = self.get_masked_patch_token(
            mask_probability=self.config['mask_ratio_img'],
            visual_token=visual_token
        )
        
        vfm_img_feat = torch.tensor(np.load(self.clip_img_feat_path + '{id}.npy'.format(id=id)), dtype=torch.float32)
            
        align_label = self.get_align_label(
            poi_name_token_ids=poi_name_token_ids,
            grid_level_pos_id=grid_level_pos_ids, 
            poi_unmask=poi_unmask, 
            img_mask=img_mask,
        )
        
        vlfm_img_feat = vfm_img_feat
        
        return {
            'poi_name_token_ids_masked': poi_name_token_ids_masked,
            'attn_mask_poi': attn_mask_poi, 
            'word_level_pos_ids': word_level_pos_ids, 
            'poi_level_pos_ids': poi_level_pos_ids, 
            'grid_level_pos_ids': grid_level_pos_ids, 
            'poi_cate_ids': poi_cate_ids, 
            'poi_masked_label': poi_masked_label, 
            'llm_poi_feat': llm_poi_feat,
            'img': img, 
            'img_mask': img_mask, 
            'img_masked_label': img_masked_label, 
            'align_label': align_label,
            'vfm_img_feat': vfm_img_feat,
            'vlfm_img_feat': vlfm_img_feat,
        }


    def __len__(self):
        return self.dataset_len


