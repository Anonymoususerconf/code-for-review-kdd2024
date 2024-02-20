import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from einops.layers.torch import Rearrange
from functools import partial
from utils import * 
import utils_dist
from MoGETransformer import *
from collections import Counter


class PreTrainedModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config['initializer_range'])
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config['initializer_range'])
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.Parameter):
            module.data.normal_(mean=0.0, std=self.config['initializer_range'])
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class POIEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.seq_len_poi = config['max_len_token']
        self.num_grid = config['num_grid_x'] * config['num_grid_y']

        self.word_embedding_table = nn.Embedding(config['vocab_size'], config['hidden_size'], padding_idx=0)
        self.word_level_pos_embedding_table = nn.Embedding(config['max_len_token'], config['hidden_size'])        
        self.poi_level_pos_embedding_table = nn.Embedding(config['max_len_poi'], config['hidden_size'])
        self.grid_level_pos_embedding_table = nn.Embedding(self.num_grid + 2, config['hidden_size'])
        self.poi_cate_embedding_table = nn.Embedding(config['poi_cate_num'], config['hidden_size'])

        self.LayerNorm = nn.LayerNorm(config['hidden_size'], eps=config['layer_norm_eps'])
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])

    def forward(self, poi_name_token_ids, word_level_pos_ids, poi_level_pos_ids, grid_level_pos_ids, poi_cate_ids):
        poi_name_token_embedding = self.word_embedding_table(poi_name_token_ids)
        word_level_pos_embedding = self.word_level_pos_embedding_table(word_level_pos_ids)
        poi_level_pos_embedding = self.poi_level_pos_embedding_table(poi_level_pos_ids)
        grid_level_pos_embedding = self.grid_level_pos_embedding_table(grid_level_pos_ids)
        poi_cate_embedding = self.poi_cate_embedding_table(poi_cate_ids)

        poi_embedding = \
            poi_name_token_embedding + \
            word_level_pos_embedding + \
            poi_level_pos_embedding + \
            grid_level_pos_embedding + \
            poi_cate_embedding

        poi_embedding = self.LayerNorm(poi_embedding)
        poi_embedding = self.dropout(poi_embedding)
        return poi_embedding
    

class SateEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        image_height, image_width = pair(config['image_size'])
        patch_height, patch_width = pair(config['patch_size'])
        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
        'Image dimensions must be divisible by the patch size.'
        self.num_patch = (image_height // patch_height) * (image_width // patch_width)
        self.seq_len_img = self.num_patch + 1
        self.patch_dim = 3 * patch_height * patch_width

        self.img2patch = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width)
        self.patch_to_emb = nn.Linear(self.patch_dim, config['hidden_size'])
        self.pos_1d_embedding_table = nn.Embedding(self.seq_len_img, config['hidden_size'])

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config['hidden_size']))
        self.mask_token = nn.Parameter(torch.zeros(config['hidden_size']))

        self.LayerNorm = nn.LayerNorm(config['hidden_size'], eps=config['layer_norm_eps'])
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])

        self.initialize_weights()


    def initialize_weights(self):
        self.cls_token.data.normal_(mean=0.0, std=self.config['initializer_range'])
        self.mask_token.data.normal_(mean=0.0, std=self.config['initializer_range'])


    def forward(self, img, mask, masking):
        patch = self.img2patch(img)
        patch_embedding = self.patch_to_emb(patch)

        if masking:
            assert mask is not None
            patch_embedding[mask] = self.mask_token

        batch_size = patch_embedding.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        patch_embedding = torch.cat([cls_tokens, patch_embedding], dim=1)

        pos_1d = torch.arange(0, patch_embedding.size(1), device=patch_embedding.device).unsqueeze(0).expand(batch_size, -1)
        pos_embedding_1d = self.pos_1d_embedding_table(pos_1d)
        patch_embedding = patch_embedding + pos_embedding_1d

        patch_embedding = self.LayerNorm(patch_embedding)
        patch_embedding = self.dropout(patch_embedding)

        return patch_embedding


class ModalEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mod_embed_tabel = nn.Embedding(2, config['hidden_size'])
    
    def forward(self, poi_embedding, img_embedding):
        batch_size = poi_embedding.size(0)
        seq_len_poi = poi_embedding.size(1)
        seq_len_img = img_embedding.size(1)
        device = poi_embedding.device

        mod_id_poi = torch.full([batch_size, seq_len_poi], 0, dtype=torch.long, device=device)
        mod_embedding_poi = self.mod_embed_tabel(mod_id_poi)
        poi_embedding = poi_embedding + mod_embedding_poi

        mod_id_img = torch.full([batch_size, seq_len_img], 1, dtype=torch.long, device=device)
        mod_embedding_img = self.mod_embed_tabel(mod_id_img)
        img_embedding = img_embedding + mod_embedding_img

        return poi_embedding, img_embedding


class ReFound(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.poi_embed_module = POIEmbed(config)
        self.img_embed_module = SateEmbed(config)
        self.mod_embed_module = ModalEmbed(config)
        self.transformer = MoGEEncoder(config)


    def prepare_poi_data(self, poi_data, masking_poi):

        if masking_poi:
            poi_name_token_ids = poi_data['poi_name_token_ids_masked']
        else:
            poi_name_token_ids = poi_data['poi_name_token_ids']

        attn_mask_poi = poi_data['attn_mask_poi']
        word_level_pos_ids = poi_data['word_level_pos_ids']
        poi_level_pos_ids = poi_data['poi_level_pos_ids']
        grid_level_pos_ids = poi_data['grid_level_pos_ids']
        poi_cate_ids = poi_data['poi_cate_ids']
            

        prepared_poi_data = {
            'poi_name_token_ids': poi_name_token_ids,
            'attn_mask_poi': attn_mask_poi,
            'word_level_pos_ids': word_level_pos_ids,
            'poi_level_pos_ids': poi_level_pos_ids,
            'grid_level_pos_ids': grid_level_pos_ids,
            'poi_cate_ids': poi_cate_ids,
        }
        return prepared_poi_data


    def prepare_img_data(self, img_data, masking_img):

        seq_len_img = self.img_embed_module.seq_len_img

        if masking_img:
            mask = img_data['img_mask']
        else:
            mask = None

        img = img_data['img']
        batch_size, device = img.size(0), img.device
        attn_mask_img = torch.ones((batch_size, seq_len_img), dtype=torch.long, device=device)
            
        prepared_img_data = {
            'img': img,
            'mask': mask,
            'attn_mask_img': attn_mask_img,
        }
        return prepared_img_data


    def forward(
        self, 
        poi_data, 
        img_data, 
        masking_poi, 
        masking_img
    ):     

        prepared_poi_data = self.prepare_poi_data(
            poi_data=poi_data, 
            masking_poi=masking_poi, 
        )

        poi_embedding = self.poi_embed_module(
            poi_name_token_ids=prepared_poi_data['poi_name_token_ids'],
            word_level_pos_ids=prepared_poi_data['word_level_pos_ids'],
            poi_level_pos_ids=prepared_poi_data['poi_level_pos_ids'],
            grid_level_pos_ids=prepared_poi_data['grid_level_pos_ids'],
            poi_cate_ids=prepared_poi_data['poi_cate_ids'],
        )
        attn_mask_poi = prepared_poi_data['attn_mask_poi']


        prepared_img_data = self.prepare_img_data(
            img_data=img_data, 
            masking_img=masking_img, 
        )

        img_embedding = self.img_embed_module(
            img=prepared_img_data['img'],
            mask=prepared_img_data['mask'],
            masking=masking_img,
        )
        attn_mask_img = prepared_img_data['attn_mask_img']


        poi_embedding, img_embedding = self.mod_embed_module(poi_embedding, img_embedding)
        all_embedding = torch.cat([poi_embedding, img_embedding], dim=1)
        attn_mask = torch.cat([attn_mask_poi, attn_mask_img], dim=1)
        split_idx = poi_embedding.shape[1]
        assert split_idx == self.poi_embed_module.seq_len_poi

        extended_attn_mask = get_extended_attention_mask(attn_mask)

        encoder_output = self.transformer(
            hidden_states=all_embedding,
            attention_mask=extended_attn_mask,
            split_idx=split_idx,
        )

        return encoder_output


class MaskedPOIPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.act_fn = nn.GELU()
        self.LayerNorm = nn.LayerNorm(config['hidden_size'], eps=config['layer_norm_eps'])
        self.decoder = nn.Linear(config['hidden_size'], config['vocab_size'], bias=False)
        self.bias = nn.Parameter(torch.zeros(config['vocab_size']))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class MaskedIMGTokenPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.act_fn = nn.GELU()
        self.LayerNorm = nn.LayerNorm(config['hidden_size'], eps=config['layer_norm_eps'])
        self.decoder = nn.Linear(config['hidden_size'], config['visual_vocab_size'])

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class SpatialAlignmentHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.act_fn = nn.GELU()
        self.LayerNorm = nn.LayerNorm(config['hidden_size'], eps=config['layer_norm_eps'])
        self.decoder = nn.Linear(config['hidden_size'], 2)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class IMGCosSimilarityHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Linear(config['hidden_size'], config['hidden_size'])
    
    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        return hidden_states


class POICosSimilarityHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Linear(config['hidden_size'], config['hidden_size'])
    
    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        return hidden_states



class ModelForPretraining(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = ReFound(config)

        self.mgdm_poi_head = MaskedPOIPredictionHead(config)
        self.mgdm_img_head = MaskedIMGTokenPredictionHead(config)
        self.cmsa_poi_head = SpatialAlignmentHead(config)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        
        self.poi_proj_for_dlfm = POICosSimilarityHead(config)
        self.img_proj_for_dvfm = IMGCosSimilarityHead(config)
        self.cos_sim = nn.CosineSimilarity(dim=1)

        self.poi_proj_for_dvlfm = POICosSimilarityHead(config)
        self.img_proj_for_dvlfm = IMGCosSimilarityHead(config)
        self.world_size = utils_dist.get_world_size()
        self.logit_scale = 1.0 / config['dvlfm_temp']
        
        self.apply(self._init_weights)


    def compute_masked_modeling_poi(self, encoder_output, poi_masked_label, seq_len_poi):
        encoder_output_poi = encoder_output[:, :seq_len_poi]

        prediction_scores = self.mgdm_poi_head(encoder_output_poi)
        poi_mgdm_loss = self.cross_entropy_loss(
            prediction_scores.view(-1, prediction_scores.size(-1)), 
            poi_masked_label.view(-1)
        )
        return poi_mgdm_loss
    
    
    def compute_masked_modeling_img(self, encoder_output, img_masked_label, num_patch):
        encoder_output_img = encoder_output[:, -num_patch:]
        prediction_scores = self.mgdm_img_head(encoder_output_img)
        img_mgdm_loss = self.cross_entropy_loss(
            prediction_scores.view(-1, prediction_scores.size(-1)), 
            img_masked_label.view(-1)
        )
        return img_mgdm_loss
    

    def compute_cross_modal_spatial_alignment(self, encoder_output, align_label, seq_len_poi):
        encoder_output_poi = encoder_output[:, :seq_len_poi]

        prediction_scores = self.cmsa_poi_head(encoder_output_poi)
        cmsa_loss = self.cross_entropy_loss(
            prediction_scores.view(-1, prediction_scores.size(-1)), 
            align_label.view(-1)
        )
        return cmsa_loss


    def compute_distillation_from_LFM(self, encoder_output, llm_poi_feat):
            
        cls_token_poi = encoder_output[:, 0]
        pool_poi_emb = cls_token_poi

        pool_poi_emb = self.poi_proj_for_dlfm(pool_poi_emb)
        pool_poi_emb_norm = pool_poi_emb / pool_poi_emb.norm(dim=1, keepdim=True)

        dlfm_loss = - self.cos_sim(pool_poi_emb_norm, llm_poi_feat).mean()
        return dlfm_loss

    
    def compute_distillation_from_VFM(self, encoder_output, vfm_img_feat, seq_len_img):

        cls_token_img = encoder_output[:, -seq_len_img]
        pool_img_emb = cls_token_img

        pool_img_emb = self.img_proj_for_dvfm(pool_img_emb)
        pool_img_emb_norm = pool_img_emb / pool_img_emb.norm(dim=1, keepdim=True)
        
        img_dst_loss = - self.cos_sim(pool_img_emb_norm, vfm_img_feat).mean()
        return img_dst_loss


    def compute_distillation_from_VLFM(self, encoder_output, vlfm_img_feat, seq_len_img):

        logit_scale = self.logit_scale

        cls_token_poi = encoder_output[:, 0]
        pool_poi_emb = cls_token_poi
        pool_poi_emb = self.poi_proj_for_dvlfm(pool_poi_emb)
        
        cls_token_img = encoder_output[:, -seq_len_img]
        pool_img_emb = cls_token_img
        pool_img_emb = self.img_proj_for_dvlfm(pool_img_emb)


        pool_poi_emb_norm = pool_poi_emb / pool_poi_emb.norm(dim=1, keepdim=True)
        pool_img_emb_norm = pool_img_emb / pool_img_emb.norm(dim=1, keepdim=True)

        if self.world_size > 1:
            gathered_vlfm_img_feat = [torch.zeros_like(vlfm_img_feat) for _ in range(self.world_size)]
            dist.all_gather(gathered_vlfm_img_feat, vlfm_img_feat)
            all_vlfm_img_feat = torch.cat(gathered_vlfm_img_feat, dim=0)
        
            gathered_pool_img_emb_norm = [torch.zeros_like(pool_img_emb_norm) for _ in range(self.world_size)]
            dist.all_gather(gathered_pool_img_emb_norm, pool_img_emb_norm)
            all_pool_img_emb_norm = torch.cat(gathered_pool_img_emb_norm, dim=0)

            gathered_pool_poi_emb_norm = [torch.zeros_like(pool_poi_emb_norm) for _ in range(self.world_size)]
            dist.all_gather(gathered_pool_poi_emb_norm, pool_poi_emb_norm)
            all_pool_poi_emb_norm = torch.cat(gathered_pool_poi_emb_norm, dim=0)

            logits_poi2img_teature = logit_scale * vlfm_img_feat @ all_vlfm_img_feat.t()
            logits_poi2img_student = logit_scale * pool_poi_emb_norm @ all_pool_img_emb_norm.t()

            logits_img2poi_teature = logits_poi2img_teature
            logits_img2poi_student = logit_scale * pool_img_emb_norm @ all_pool_poi_emb_norm.t()

        else:
            logits_poi2img_teature = logit_scale * vlfm_img_feat @ vlfm_img_feat.t()
            logits_poi2img_student = logit_scale * pool_poi_emb_norm @ pool_img_emb_norm.t()

            logits_img2poi_teature = logits_poi2img_teature
            logits_img2poi_student = logits_poi2img_student.t()

        
        poi2img_dvlfm_loss = F.kl_div(
            F.log_softmax(logits_poi2img_student, dim=-1), 
            F.softmax(logits_poi2img_teature, dim=-1), 
            reduction='batchmean'
        )
        img2poi_dvlfm_loss = F.kl_div(
            F.log_softmax(logits_img2poi_student, dim=-1), 
            F.softmax(logits_img2poi_teature, dim=-1), 
            reduction='batchmean'
        )
        dvlfm_loss = (poi2img_dvlfm_loss + img2poi_dvlfm_loss) * 0.5
        return dvlfm_loss



    def forward(
        self, 
        poi_data, 
        img_data, 
        poi_masked_label, 
        img_masked_label,
        align_label,
        llm_poi_feat,
        vfm_img_feat,
        vlfm_img_feat,
    ):

        loss_dict = {
            'poi_mgdm_loss': 0,  'img_mgdm_loss': 0, 'cmsa_loss': 0,
            'dlfm_loss': 0, 'dvfm_loss': 0, 'dvlfm_loss': 0
        }

        encoder_output = self.encoder(
            poi_data=poi_data, 
            img_data=img_data, 
            masking_poi=True,
            masking_img=True,
        )
        
        poi_mgdm_loss = self.compute_masked_modeling_poi(
            encoder_output=encoder_output,
            poi_masked_label=poi_masked_label,
            seq_len_poi=self.encoder.poi_embed_module.seq_len_poi
        )
        loss_dict['poi_mgdm_loss'] = poi_mgdm_loss

        
        img_mgdm_loss = self.compute_masked_modeling_img(
            encoder_output=encoder_output,
            img_masked_label=img_masked_label,
            num_patch=self.encoder.img_embed_module.num_patch
        )
        loss_dict['img_mgdm_loss'] = img_mgdm_loss
        

        cmsa_loss = self.compute_cross_modal_spatial_alignment(
            encoder_output=encoder_output, 
            align_label=align_label, 
            seq_len_poi=self.encoder.poi_embed_module.seq_len_poi
        )
        loss_dict['cmsa_loss'] = cmsa_loss
        
        
        dlfm_loss = self.compute_distillation_from_LFM(
            encoder_output=encoder_output,
            llm_poi_feat=llm_poi_feat,
        )
        loss_dict['dlfm_loss'] = dlfm_loss
        
        
        dvfm_loss = self.compute_distillation_from_VFM(
            encoder_output=encoder_output, 
            vfm_img_feat=vfm_img_feat,
            seq_len_img=self.encoder.img_embed_module.seq_len_img,
        )
        loss_dict['dvfm_loss'] = dvfm_loss
        

        dvlfm_loss = self.compute_distillation_from_VLFM(
            encoder_output=encoder_output, 
            vlfm_img_feat=vlfm_img_feat,
            seq_len_img=self.encoder.img_embed_module.seq_len_img,
        )
        loss_dict['dvlfm_loss'] = dvlfm_loss

        return loss_dict




class AttnPool(nn.Module):
    def __init__(self, config, hidden_size=32):
        super(AttnPool, self).__init__()

        self.l1 = nn.Linear(config['hidden_size'], hidden_size, bias=True)
        self.ac = nn.Tanh()
        self.l2 = nn.Linear(int(hidden_size), 1, bias=False)        

    def forward(self, z):
        w = self.l1(z)
        w = self.ac(w)
        w = self.l2(w)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1)
        


class UrbanVillageDetectionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(config['fdrop'])
        self.decoder = nn.Linear(config['hidden_size'], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden_states):
        output = self.dense(hidden_states)
        output = self.act_fn(output)
        output = self.dropout(output)
        output = self.decoder(output)
        output = self.sigmoid(output)
        return output



class FinetuneUrbanVillageDetection(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.encoder = ReFound(config)
        self.target_prediction = UrbanVillageDetectionHead(config)

        if config['agg'] == 'attn':
            self.attn_agg = AttnPool(config)

        self.apply(self._init_weights)

    def forward(self, poi_data, img_data):

        encoder_output = self.encoder(
            poi_data=poi_data, 
            img_data=img_data, 
            masking_poi=False,
            masking_img=False,
        )

        if self.config['agg'] == 'avr':
            seq_len_img = self.encoder.img_embed_module.seq_len_img
            cls_output_poi = encoder_output[:, 0]
            cls_output_img = encoder_output[:, -seq_len_img]
            agg_output = 0.5 * (cls_output_poi + cls_output_img)
        
        elif self.config['agg'] == 'attn':
            seq_len_img = self.encoder.img_embed_module.seq_len_img
            cls_output_poi = encoder_output[:, 0]
            cls_output_img = encoder_output[:, -seq_len_img]
            agg_output = torch.stack([cls_output_poi, cls_output_img], dim=1)
            agg_output = self.attn_agg(agg_output)
        
        prediction_scores = self.target_prediction(agg_output)
        return prediction_scores



class CommercialActivenessPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(config['fdrop'])
        self.decoder = nn.Linear(config['hidden_size'], 1)

    def forward(self, hidden_states):
        output = self.dense(hidden_states)
        output = self.act_fn(output)
        output = self.dropout(output)
        output = self.decoder(output)
        return output



class FinetuneCommercialActivenessPrediction(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.encoder = ReFound(config)
        self.target_prediction = CommercialActivenessPredictionHead(config)

        if config['agg'] == 'attn':
            self.attn_agg = AttnPool(config)

        self.apply(self._init_weights)

    def forward(self, poi_data, img_data):
        
        encoder_output = self.encoder(
            poi_data=poi_data, 
            img_data=img_data, 
            masking_poi=False,
            masking_img=False,
        )

        if self.config['agg'] == 'avr':
            seq_len_img = self.encoder.img_embed_module.seq_len_img
            cls_output_poi = encoder_output[:, 0]
            cls_output_img = encoder_output[:, -seq_len_img]
            agg_output = 0.5 * (cls_output_poi + cls_output_img)     
        
        elif self.config['agg'] == 'attn':
            seq_len_img = self.encoder.img_embed_module.seq_len_img
            cls_output_poi = encoder_output[:, 0]
            cls_output_img = encoder_output[:, -seq_len_img]
            agg_output = torch.stack([cls_output_poi, cls_output_img], dim=1)
            agg_output = self.attn_agg(agg_output)
        
        prediction_scores = self.target_prediction(agg_output)
        return prediction_scores




class PopulationPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.dense = nn.Linear(config['hidden_size'], config['hidden_size'])

        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(config['fdrop'])

        self.decoder = nn.Linear(config['hidden_size'], 1)

    def forward(self, hidden_states):
        output = self.dense(hidden_states)
        output = self.act_fn(output)
        output = self.dropout(output)
        output = self.decoder(output)
        return output



class FinetunePopulationPredictionHead(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.encoder = ReFound(config)
        self.target_prediction = PopulationPredictionHead(config)

        if config['agg'] == 'attn':
            self.attn_agg = AttnPool(config)
        
        self.apply(self._init_weights)

    def forward(self, poi_data, img_data):
        
        encoder_output = self.encoder(
            poi_data=poi_data, 
            img_data=img_data, 
            masking_poi=False,
            masking_img=False,
        )

        if self.config['agg'] == 'avr':
            seq_len_img = self.encoder.img_embed_module.seq_len_img
            cls_output_poi = encoder_output[:, 0]
            cls_output_img = encoder_output[:, -seq_len_img]
            agg_output = 0.5 * (cls_output_poi + cls_output_img)
        
        elif self.config['agg'] == 'attn':
            seq_len_img = self.encoder.img_embed_module.seq_len_img
            cls_output_poi = encoder_output[:, 0]
            cls_output_img = encoder_output[:, -seq_len_img]
            agg_output = torch.stack([cls_output_poi, cls_output_img], dim=1)
            agg_output = self.attn_agg(agg_output)
        
        prediction_scores = self.target_prediction(agg_output)
        return prediction_scores




class FeatureExtractor(PreTrainedModel):
    # extract region representation for feature-based prediction
    def __init__(self, config):
        super().__init__(config)

        self.encoder = ReFound(config)
        self.apply(self._init_weights)

    def forward(self, poi_data, img_data):

        encoder_output = self.encoder(
            poi_data=poi_data, 
            img_data=img_data, 
            masking_poi=False,
            masking_img=False,
        )

        seq_len_img = self.encoder.img_embed_module.seq_len_img
        cls_output_poi = encoder_output[:, 0]
        cls_output_img = encoder_output[:, -seq_len_img]
        cls_output = torch.stack([cls_output_poi, cls_output_img], dim=1)

        return cls_output



class FeatureBasedUrbanVillageDetection(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.target_prediction = UrbanVillageDetectionHead(config)

        if config['agg'] == 'attn':
            self.attn_agg = AttnPool(config)

        self.apply(self._init_weights)

    def forward(self, region_emb):

        if self.config['agg'] == 'avr':
            cls_output_poi = region_emb[:, 0]
            cls_output_img = region_emb[:, 1]
            agg_output = 0.5 * (cls_output_poi + cls_output_img)

        elif self.config['agg'] == 'attn':
            cls_output_poi = region_emb[:, 0]
            cls_output_img = region_emb[:, 1]
            agg_output = torch.stack([cls_output_poi, cls_output_img], dim=1)
            agg_output = self.attn_agg(agg_output)

        prediction_scores = self.target_prediction(agg_output)

        return prediction_scores




class FeatureBasedCommercialActivenessPrediction(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.target_prediction = CommercialActivenessPredictionHead(config)

        if config['agg'] == 'attn':
            self.attn_agg = AttnPool(config)

        self.apply(self._init_weights)

    def forward(self, region_emb):

        if self.config['agg'] == 'avr':
            cls_output_poi = region_emb[:, 0]
            cls_output_img = region_emb[:, 1]
            agg_output = 0.5 * (cls_output_poi + cls_output_img)

        elif self.config['agg'] == 'attn':
            cls_output_poi = region_emb[:, 0]
            cls_output_img = region_emb[:, 1]
            agg_output = torch.stack([cls_output_poi, cls_output_img], dim=1)
            agg_output = self.attn_agg(agg_output)

        prediction_scores = self.target_prediction(agg_output)

        return prediction_scores





class FeatureBasedPopulationPrediction(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.target_prediction = PopulationPredictionHead(config)

        if config['agg'] == 'attn':
            self.attn_agg = AttnPool(config)

        self.apply(self._init_weights)

    def forward(self, region_emb):

        if self.config['agg'] == 'avr':
            cls_output_poi = region_emb[:, 0]
            cls_output_img = region_emb[:, 1]
            agg_output = 0.5 * (cls_output_poi + cls_output_img)

        elif self.config['agg'] == 'attn':
            cls_output_poi = region_emb[:, 0]
            cls_output_img = region_emb[:, 1]
            agg_output = torch.stack([cls_output_poi, cls_output_img], dim=1)
            agg_output = self.attn_agg(agg_output)

        prediction_scores = self.target_prediction(agg_output)

        return prediction_scores



