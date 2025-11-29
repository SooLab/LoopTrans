from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dino import vision_transformer as vits
from models.dino.utils import load_pretrained_weights
from models.model_util import *


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):   
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice


class Net(nn.Module):

    def __init__(self, cluster, aff_classes=36):
        super(Net, self).__init__()

        # --- hyper-parameters --- #
        self.noise_classes = 15
        self.clu_num = 4
        self.fg_threshold = 0.92

        # --- dino-vit features --- #
        self.vit_feat_dim = 384
        self.patch = 16
        self.stride = 16

        self.aff_classes = aff_classes
        self.class_num = aff_classes + self.noise_classes
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.vit_model = vits.__dict__['vit_small'](patch_size=self.patch, num_classes=0, stride=self.stride)
        load_pretrained_weights(self.vit_model, '', None, 'vit_small', self.patch)

        self.cluster = cluster

        # --- learning parameters --- #
        self.aff_proj = Mlp(in_features=self.vit_feat_dim, hidden_features=int(self.vit_feat_dim * 4),
                            act_layer=nn.GELU, drop=0.)
        self.pixel_dec = nn.Sequential(
            nn.Conv2d(self.vit_feat_dim, self.vit_feat_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.vit_feat_dim),
            nn.ReLU(True),
            nn.Conv2d(self.vit_feat_dim, self.vit_feat_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.vit_feat_dim),
            nn.ReLU(True),
        )
        self.scam = nn.Sequential(
            nn.Conv2d(self.vit_feat_dim, self.vit_feat_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.vit_feat_dim),
            nn.ReLU(True),
            nn.Conv2d(self.vit_feat_dim, self.vit_feat_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.vit_feat_dim),
            nn.ReLU(True),
        )
        self.aff_fc = nn.Conv2d(self.vit_feat_dim, self.class_num, 1)

        # --- loss functions --- #
        self.mse_loss = nn.MSELoss()
        self.dice_loss = DiceLoss()

    def forward(self, exo, ego, aff_label, epoch):
        num_exo = exo.shape[1]
        exo = exo.flatten(0, 1)  # b*num_exo x 3 x 224 x 224

        # --- Extract deep descriptors from DINO-vit --- #
        with torch.no_grad():
            _, ego_key, ego_attn = self.vit_model.get_last_key(ego)
            _, exo_key, _ = self.vit_model.get_last_key(exo)
            ego_desc = ego_key.permute(0, 3, 1, 2).flatten(1, 2)[:, :, 1:].transpose(-2, -1).detach()
            exo_desc = exo_key.permute(0, 3, 1, 2).flatten(1, 2)[:, :, 1:].transpose(-2, -1).detach()
        b, hw, c = ego_desc.shape
        h = int(sqrt(hw))
        w = int(sqrt(hw))
        ego_proj = (ego_desc + self.aff_proj(ego_desc)).transpose(-2, -1).reshape(b, c, h, w).contiguous()
        exo_proj = (exo_desc + self.aff_proj(exo_desc)).transpose(-2, -1).reshape(b*num_exo, c, h, w).contiguous()
        ego_desc = ego_desc.transpose(-2, -1).reshape(b, c, h, w).contiguous()
        exo_desc = exo_desc.transpose(-2, -1).reshape(b*num_exo, c, h, w).contiguous()
        exo_desc_re = exo_desc.reshape(b, num_exo, c, h, w)

        ego_cls_attn = ego_attn[:, :, 0, 1:].reshape(b, 6, h, w)
        head_idxs = [0, 1, 3]
        ego_sam = ego_cls_attn[:, head_idxs].mean(1)
        ego_sam = normalize_minmax(ego_sam)
        ego_sam_flat = ego_sam.flatten(-2, -1) # b x hw

        # --- Affordance CAM generation --- #
        exo_proj = self.scam(exo_proj)
        aff_cam = self.aff_fc(exo_proj)  # b*num_exo x num_class x h x w
        aff_logits = self.gap(aff_cam).reshape(b, num_exo, self.class_num)
        aff_cam_re = aff_cam.reshape(b, num_exo, self.class_num, h, w)
        gt_aff_cam = torch.zeros(b, num_exo, h, w).cuda()
        for b_ in range(b):
            gt_aff_cam[b_, :] = normalize_minmax(aff_cam_re[b_, :, aff_label[b_]])
            

        ego_proj_1 = self.pixel_dec(ego_proj)
        ego_pred = self.aff_fc(ego_proj_1) # b x num_class x h x w
        aff_logits_ego = self.gap(ego_pred).view(b, self.class_num)

        gt_ego_cam = torch.zeros(b, h, w).cuda()
        loss_con = torch.zeros(1).cuda()
        for b_ in range(b):
            gt_ego_cam[b_] = ego_pred[b_, aff_label[b_]]
            loss_con += concentration_loss(ego_pred[b_])
        gt_ego_cam = normalize_minmax(gt_ego_cam) # b x h x w
        loss_con /= b

        # --- correlation loss --- #
        ego_aff_feat = ego_pred.flatten(-2, -1)
        exo_aff_feat = aff_cam_re.transpose(1, 2).flatten(-2, -1).flatten(-2, -1)
        ego_aff_corr = torch.matmul(ego_aff_feat, ego_aff_feat.transpose(1, 2))
        exo_aff_corr = torch.matmul(exo_aff_feat, exo_aff_feat.transpose(1, 2))
        loss_corr = 1 - F.cosine_similarity(ego_aff_corr, exo_aff_corr).mean()

        # --- forward in the scam branch with the ego image --- #
        cross_proj = self.scam(ego_proj) # b x c x h x w
        cross_cam = self.aff_fc(cross_proj) # b x num_class x h x w
        cross_logits = self.gap(cross_cam).view(b, self.class_num)
        loss_cls = nn.CrossEntropyLoss().cuda()(cross_logits, aff_label)
        gt_cross_cam = torch.zeros(b, h, w).cuda()
        for b_ in range(b):
            gt_cross_cam[b_] = cross_cam[b_, aff_label[b_]]
        gt_cross_cam = normalize_minmax(gt_cross_cam)

        # --- denoising distillation loss --- #
        loss_denoise = torch.zeros(1).cuda()
        for b_ in range(b):
            aff_query = (aff_cam_re[b_, :, aff_label[b_]].unsqueeze(1) * exo_desc_re[b_]).mean(0).mean(-1).mean(-1) # c
            aff_query = F.normalize(aff_query, p=2, dim=-1)
            pos_key = (ego_pred[b_, aff_label[b_]].unsqueeze(0) * ego_desc[b_]).mean(-1).mean(-1) # c
            neg_keys = (aff_cam_re[b_, :, self.aff_classes:].unsqueeze(2) * exo_desc_re[b_].unsqueeze(1)).mean(0).mean(-1).mean(-1) # num_neg x c
            keys = torch.cat((pos_key.unsqueeze(0), neg_keys), dim=0) # num_neg+1 x c
            keys = F.normalize(keys, p=2, dim=-1)
            aff_sim = torch.matmul(keys, aff_query.unsqueeze(-1)).squeeze(-1) # num_neg+1
            loss_denoise += -torch.log(torch.exp(aff_sim[0]) / torch.exp(aff_sim).sum())
        loss_denoise /= b     
        # --- pixel loss --- #
        pseudo_labels = torch.zeros(b, h, w).cuda()
        loss_mse = torch.zeros(1).cuda()
        loss_dice = torch.zeros(1).cuda()

        num_clu = self.clu_num
        ego_desc_re = ego_desc.flatten(-2, -1).transpose(1, 2) # b x hw x c
        similarity = self.cluster(ego_desc_re) # b x hw x num_token
        labels = torch.argmax(similarity, dim=-1) # b x hw
        cam_mask = gt_cross_cam.flatten(-2, -1) > self.fg_threshold # b x hw
        for b_ in range(b):
            part_scores = torch.zeros(num_clu).cuda()
            part_sam = torch.zeros(num_clu).cuda()
            for part in range(num_clu):
                part_mask = (labels[b_] == part)
                inter_mask = cam_mask[b_] & part_mask
                union_mask = cam_mask[b_] | part_mask
                part_scores[part] = inter_mask.sum() / (union_mask.sum() + 1e-6)
                part_sam[part] = ego_sam_flat[b_][part_mask].mean()
            bg_part = torch.argmin(part_sam)
            part_scores[bg_part] = -1
            aff_part = torch.argmax(part_scores)
            pseudo_label = (labels[b_] == aff_part).float().reshape(h, w)
            pseudo_labels[b_] = pseudo_label
            loss_mse += self.mse_loss(gt_ego_cam[b_], pseudo_label)
            loss_dice += self.dice_loss(gt_ego_cam[b_], pseudo_label)
        loss_mse /= b
        loss_dice /= b

        # --- return --- #

        masks = {'exo_aff': gt_aff_cam, 'ego_sam': ego_sam}
        logits = {'aff': aff_logits, 'aff_ego': aff_logits_ego}

        return masks, logits, loss_cls, loss_mse, loss_dice, loss_corr, loss_denoise, loss_con

    @torch.no_grad()
    def test_forward(self, ego, aff_label):
        _, ego_key, ego_attn = self.vit_model.get_last_key(ego)
        ego_desc = ego_key.permute(0, 3, 1, 2).flatten(1, 2)[:, :, 1:].transpose(-2, -1).contiguous()
        b, hw, c = ego_desc.shape
        h = int(sqrt(hw))
        w = int(sqrt(hw))

        ego_proj = (ego_desc + self.aff_proj(ego_desc)).transpose(-2, -1).reshape(b, c, h, w).contiguous()
        ego_proj = self.pixel_dec(ego_proj)
        ego_pred = self.aff_fc(ego_proj)

        gt_ego_cam = torch.zeros(b, h, w).cuda()
        for b_ in range(b):
            gt_ego_cam[b_] = ego_pred[b_, aff_label[b_]]
            # gt_ego_cam[b_] = ego_pred[b_, self.aff_classes]

        # optional
        # norm_cam = normalize_minmax(gt_ego_cam)
        # mask = (norm_cam > 0.2).float()
        # gt_ego_cam = mask * gt_ego_cam + gt_ego_cam

        return gt_ego_cam