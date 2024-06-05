# Copyright (c) 2023 Yanhao Li
# yanhao.li@outlook.com

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.


from typing import List

import torch
from torch import nn, optim
import numpy as np
import pytorch_lightning as pl
import torchmetrics
from torchmetrics.aggregation import MeanMetric

from .third_party.SyntheticImagesAnalysis.DnCNN import make_net


class MonotoneLayer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.a = torch.tensor(0.)
        self.a = nn.Parameter(self.a)
        self.b = torch.tensor(0.)
        self.b = nn.Parameter(self.b)
        
    def forward(self, x):
        x = torch.exp(self.a) * x + self.b
        x = torch.sigmoid(x)
        return x


def get_model():
    num_levels = 17
    out_channel = 3
    model = make_net(3, kernels=[3, ] * num_levels,
                        features=[64, ] * (num_levels - 1) + [out_channel],
                        bns=[False, ] + [True, ] *
                        (num_levels - 2) + [False, ],
                        acts=['relu', ] * (num_levels - 1) + ['linear', ],
                        dilats=[1, ] * num_levels,
                        bn_momentum=0.1, padding=0)
    weights_path = "src/third_party/SyntheticImagesAnalysis/DenoiserWeight/model_best.th"
    state_dict = torch.load(weights_path, torch.device('cpu'))
    model.load_state_dict(state_dict["network"])
    model.eval()

    return model

def single2tensor4(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(0, 3, 1, 2).float().unsqueeze(0)


class DnCNN(nn.Module):
    def __init__(self, freeze:bool=True) -> None:
        super().__init__()
        self.model = get_model()
        self.freeze = freeze
    
    def forward(self, imgs):
        """
        Parameters
        ----------
        imgs : BxCxHxW

        Returns
        -------
        residuals : BxCxHxW
        """
        if self.freeze:
            with torch.no_grad():
                img_residuals: torch.Tensor = self.model(imgs)
        else:
           img_residuals: torch.Tensor = self.model(imgs)

        return img_residuals


class MaskSim(pl.LightningModule):
    def __init__(self, img_size:int, channels:int, lr:float=1e-2, kernel_sz:int=3,
                 preprocess:str="linear3x3",
                 num_masks:int=1, 
                 reference_pattern:np.ndarray=None,
                 preproc_freeze:bool=True
                 ):
        super().__init__()

        self.save_hyperparameters()

        self.img_size = img_size
        self.channels = channels
        self.fft = True

        self.preproc = None
        if preprocess == "cross_diff":
            self.init_cross_diff()
            self.preproc = self.preprocess_cross_diff
        elif preprocess == "DnCNN":
            self.preproc = DnCNN(freeze=preproc_freeze)
            if preproc_freeze:
                self.preproc.requires_grad_(False)
            else:
                self.preproc.requires_grad_(True)
        elif preprocess == "linear3x3":
            self.preproc = nn.Sequential(
                nn.Conv2d(in_channels=self.channels, out_channels=self.channels,
                            kernel_size=3, bias=False, padding=1),
            )
        else:
            self.preproc = nn.Identity()
        # print("\nApplying preprocessing: ", preprocess, "\n")
        
        self.ref_pattern_list = nn.ParameterList()
        self.mask_pre_activ_list = nn.ParameterList()
        for _ in range(num_masks):
            if reference_pattern is not None:
                ref_pattern = torch.from_numpy(reference_pattern)
                ref_pattern = nn.Parameter(ref_pattern, requires_grad=False)
                # print("Using fixed pattern")
            else:
                ref_pattern = torch.randn(channels, img_size, img_size) * 0.01
                # For DnCNN, the output size is 478
                if preprocess == "DnCNN":
                    # determine the output size of DnCNN
                    out_dncnn = self.preproc(torch.randn(1, channels, img_size, img_size))
                    # print("out_dncnn:", out_dncnn.shape)
                    # ref_pattern = torch.randn(channels, 478, 478) * 0.01
                    ref_pattern = torch.randn(out_dncnn.shape[1:4]) * 0.01
                ref_pattern = nn.Parameter(ref_pattern, requires_grad=True)
                ref_pattern.requires_grad_(True)
            self.ref_pattern_list.append(ref_pattern)

            # For DnCNN, the output size is 478
            mask_pre_activation = torch.randn(channels, img_size, img_size) * 0.01
            if preprocess == "DnCNN":
                out_dncnn = self.preproc(torch.randn(1, channels, img_size, img_size))
                # print("out_dncnn:", out_dncnn.shape)
                # mask_pre_activation = torch.randn(channels, 478, 478) * 0.01
                mask_pre_activation = torch.randn(out_dncnn.shape[1:4]) * 0.01
            mask_pre_activation = nn.Parameter(mask_pre_activation)
            # restrain mask to [0, 1] and sum = 1
            mask_pre_activation.requires_grad_(True)
            self.mask_pre_activ_list.append(mask_pre_activation)

        
        assert kernel_sz % 2 == 1, f"kernel_sz = {kernel_sz}"
        self.mask_conv = nn.Conv2d(in_channels=self.channels, out_channels=self.channels,
                                   groups=self.channels, kernel_size=kernel_sz, bias=False, padding=(kernel_sz - 1) // 2)
        weights = torch.ones((kernel_sz, kernel_sz))
        weights = weights.view(1, 1, kernel_sz, kernel_sz).repeat(self.channels, 1, 1, 1) / (kernel_sz ** 2)
        self.mask_conv.weight = nn.Parameter(weights)
        self.mask_conv.requires_grad_(False)


        self.in_conv_fft = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels,
                        kernel_size=1, bias=False, padding=0),
        )

        self.norm_vector = nn.BatchNorm1d(self.channels, affine=True)  # maybe no batchnorm is better
        # self.norm_vector = nn.BatchNorm1d(self.channels, affine=False)  # maybe not affine but simply y=kx

        self.clf_a = torch.tensor(0.)
        self.clf_a = nn.Parameter(self.clf_a)
        self.clf_b = torch.tensor(0.)
        self.clf_b = nn.Parameter(self.clf_b)

        self.lr = lr

        self.train_auroc = torchmetrics.AUROC(task="binary")
        self.valid_auroc = torchmetrics.AUROC(task="binary")
        self.test_auroc = torchmetrics.AUROC(task="binary")

        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.valid_acc = torchmetrics.Accuracy(task="binary")
        self.test_acc = torchmetrics.Accuracy(task="binary")

        self.train_success_rate = MeanMetric()
        self.valid_success_rate = MeanMetric()
        self.test_success_rate = MeanMetric()

        self.train_sim_pos = MeanMetric()
        self.train_sim_neg = MeanMetric()
        self.valid_sim_pos = MeanMetric()
        self.valid_sim_neg = MeanMetric()
        self.test_sim_pos = MeanMetric()
        self.test_sim_neg = MeanMetric()

        self.train_loss_pos = MeanMetric()
        self.train_loss_neg = MeanMetric()
        self.valid_loss_pos = MeanMetric()
        self.valid_loss_neg = MeanMetric()

    def init_cross_diff(self):
        self.conv_filter = nn.Conv2d(in_channels=self.channels, 
                                     out_channels=self.channels,
                                     groups=self.channels,
                                     kernel_size=2, bias=False, padding=0)
        weights = torch.from_numpy(np.array([[1.0, -1.0], [-1.0, 1.0]]))
        weights = weights.view(1, 1, 2, 2).repeat(self.channels, 1, 1, 1)
        self.conv_filter.weight = nn.Parameter(weights)
        self.conv_filter.requires_grad_(False)

    def preprocess_cross_diff(self, imgs:torch.Tensor) -> torch.Tensor:
        """ imgs: BxCxHxW
            return: BxCxHxW
        """
        imgs_r = torch.zeros_like(imgs)
        imgs_r[:, :, :-1, :-1] = self.conv_filter(imgs)

        return imgs_r

    def get_mask(self) -> List[torch.Tensor]:
        mask_list = []
        for mask_pre in self.mask_pre_activ_list:
            mask = self.mask_conv(mask_pre)
            C, H, W = mask.shape

            mask = torch.sigmoid(mask)
            mask_list.append(mask)

        return mask_list

    def compute_residual(self, imgs:torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.preproc(imgs.to(self.device)).detach().cpu()

    def compute_fft(self, imgs:torch.Tensor, is_convolved=True) -> torch.Tensor:
        imgs = self.preproc(imgs.to(self.device))
        ffts = torch.fft.fftshift(torch.fft.fft2(imgs, norm="ortho", dim=(-1, -2)), dim=(-1, -2)).abs()
        ffts = (ffts + 1e-10).log()
        if is_convolved:
            ffts = self.in_conv_fft(ffts)

        return ffts.detach().cpu()

    def forward(self, imgs) -> torch.Tensor:
        """_summary_

        Parameters
        ----------
        imgs : BxCxHxW
        """
        B, C, H, W = imgs.shape
        imgs = self.preproc(imgs)
        ffts = torch.fft.fftshift(torch.fft.fft2(imgs, norm="ortho", dim=(-1, -2)), dim=(-1, -2)).abs()
        ffts = (ffts + 1e-10).log()

        ffts = self.in_conv_fft(ffts)

        mask_list = self.get_mask()

        similarities = []
        for mask, ref_pattern in zip(mask_list, self.ref_pattern_list):
            mask = nn.Dropout(p=0.0)(mask)
            ref_pattern = nn.Dropout(p=0.0)(ref_pattern)

            mask = mask[None, ...]
            ffts_masked = ffts * mask  # B, C*n, H, W

            vectors_ref = ref_pattern.view(C, -1)  # C*n, K
            vectors = ffts_masked.view(B, C, -1)  # B, C, K

            K = vectors_ref.shape[-1]

            vectors_ref = vectors_ref - torch.mean(vectors_ref, dim=-1)[..., None] 
            vectors = self.norm_vector(vectors)

            vectors_ref = vectors_ref[None, ...].expand(B, C, K) # B, C*n, K

            similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-08)(vectors, vectors_ref)  # B, C
            similarity = similarity.mean(dim=-1)  # B
            similarities.append(similarity)
        similarities = torch.stack(similarities) # num_mask, B

        return torch.max(similarities, dim=0)[0]

    def compute_similarities(self, imgs):
        similarities = self.forward(imgs) # B
        return similarities
    
    def compute_similarity_map(self, imgs) -> torch.Tensor:
        imgs = self.preproc(imgs)
        B, C, H, W = imgs.shape
        ffts = torch.fft.fftshift(torch.fft.fft2(imgs, norm="ortho", dim=(-1, -2)), dim=(-1, -2)).abs()
        ffts = (ffts + 1e-10).log()
        ffts = self.in_conv_fft(ffts)
        mask_list = self.get_mask()

        similarities = []
        for mask, ref_pattern in zip(mask_list, self.ref_pattern_list):
            mask = nn.Dropout(p=0.0)(mask)
            ref_pattern = nn.Dropout(p=0.0)(ref_pattern)

            mask = mask[None, ...]
            ffts_masked = ffts * mask  # B, C*n, H, W

            vectors_ref = ref_pattern.view(C, -1)  # C*n, K
            vectors = ffts_masked.view(B, C, -1)  # B, C, K


            K = vectors_ref.shape[-1]

            vectors_ref = vectors_ref - torch.mean(vectors_ref, dim=-1)[..., None] 
            vectors = self.norm_vector(vectors)

            vectors_ref = vectors_ref[None, ...].expand(B, C, K) # B, C*n, K

            print("vectors", vectors.shape) # B, C, K=HW
            print("vectors_ref", vectors_ref.shape) # B, C, K

            vectors_l2 = torch.sqrt(torch.sum(vectors * vectors, dim=-1))[..., None] # B, C, 1
            vectors_ref_l2 = torch.sqrt(torch.sum(vectors_ref * vectors_ref, dim=-1))[..., None] # B, C, 1

            eps = 1e-8
            vectors = vectors / (vectors_l2 + eps)
            vectors_ref = vectors_ref / (vectors_ref_l2 + eps)
            similarity_map = (vectors * vectors_ref).reshape(B, C, H, W)
            print(similarity_map)

            return similarity_map

            # similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-08)(vectors, vectors_ref)

        # return similarities
    
    def compute_probs(self, imgs):
        similarities = self.forward(imgs) # B
        logits = torch.exp(self.clf_a) * similarities + self.clf_b
        
        probs = torch.sigmoid(logits)
        return probs

    def one_step(self, batch):

        imgs_neg, imgs_pos = batch  # B, C, H, W
        B, C, H, W = imgs_neg.shape
        
        neg_similarity = self.forward(imgs_neg)
        pos_similarity = self.forward(imgs_pos)

        return neg_similarity, pos_similarity

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        similarity = self.forward(imgs)

        similarity_pos = similarity[labels == 1]
        similarity_neg = similarity[labels == 0]
        logits_pos = torch.exp(self.clf_a) * similarity_pos + self.clf_b
        
        # original: abs negative similarity
        logits_neg = torch.exp(self.clf_a) * similarity_neg.abs() + self.clf_b
        # what about using relu?
        # logits_neg = torch.exp(self.clf_a) * torch.relu(similarity_neg) + self.clf_b
        # what about allowing negative similarities?
        # logits_neg = torch.exp(self.clf_a) * similarity_neg + self.clf_b

        probs_pos = nn.Sigmoid()(logits_pos).flatten()
        probs_neg = nn.Sigmoid()(logits_neg).flatten()
        
        # loss = 0
        # if len(probs_pos) > 0:
        #     loss += nn.BCELoss()(probs_pos, torch.ones_like(probs_pos))
        # if len(probs_neg) > 0:
        #     loss += nn.BCELoss()(probs_neg, torch.zeros_like(probs_neg))
        
        loss_ce = nn.BCELoss()(
            torch.concat([probs_pos, probs_neg]),
            torch.concat([torch.ones_like(probs_pos), torch.zeros_like(probs_neg)]),
        )
        
        loss_neg = similarity_neg.mean().abs() # force the negative similarities to be centered at 0

        lam = 10
        loss = loss_ce + lam * loss_neg

        loss = loss_ce

        logits = torch.exp(self.clf_a) * similarity + self.clf_b

        self.train_auroc.update(torch.sigmoid(logits), labels)
        self.train_acc.update(torch.sigmoid(logits), labels)

        probs = nn.Sigmoid()(logits).flatten()

        self.log('train_loss', loss.item(), on_epoch=True, sync_dist=True)

        probs_pos = probs.detach()[labels == 1]
        if len(probs_pos) > 0:
            loss_pos = nn.BCELoss()(probs_pos, torch.ones_like(probs_pos))
            self.train_loss_pos.update(loss_pos)

            pos_similarity = similarity.detach()[labels == 1]
            self.train_sim_pos.update(pos_similarity.mean())

        probs_neg = probs.detach()[labels == 0]
        if len(probs_neg) > 0:
            loss_neg = nn.BCELoss()(probs_neg, torch.zeros_like(probs_neg))
            self.train_loss_neg.update(loss_neg)

            neg_similarity = similarity.detach()[labels == 0]
            self.train_sim_neg.update(neg_similarity.mean())

        return loss
    
    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        similarity = self.forward(imgs)

        logits = torch.exp(self.clf_a) * similarity + self.clf_b

        self.valid_auroc.update(torch.sigmoid(logits), labels)
        self.valid_acc.update(torch.sigmoid(logits), labels)

        probs = nn.Sigmoid()(logits)

        loss = nn.BCELoss()(probs.flatten(), labels.float())
        
        self.log('valid_loss', loss.item(), on_epoch=True, sync_dist=True)

        probs_pos = probs.detach()[labels == 1]
        if len(probs_pos) > 0:
            loss_pos = nn.BCELoss()(probs_pos, torch.ones_like(probs_pos))
            self.valid_loss_pos.update(loss_pos)

            pos_similarity = similarity.detach()[labels == 1]
            self.valid_sim_pos.update(pos_similarity.mean())

        probs_neg = probs.detach()[labels == 0]
        if len(probs_neg) > 0:
            loss_neg = nn.BCELoss()(probs_neg, torch.zeros_like(probs_neg))
            self.valid_loss_neg.update(loss_neg)

            neg_similarity = similarity.detach()[labels == 0]
            self.valid_sim_neg.update(neg_similarity.mean())
        
    def test_step(self, batch, batch_idx):
        ffts_neg, ffts_pos = batch  # B, C, H, W

        B, C, H, W = ffts_neg.shape

        neg_similarity, pos_similarity = self.one_step(batch)

        self.test_sim_neg.update(neg_similarity.mean())
        self.test_sim_pos.update(pos_similarity.mean())

        neg_logits = torch.exp(self.clf_a) * neg_similarity[..., None] + self.clf_b
        pos_logits = torch.exp(self.clf_a) * pos_similarity[..., None] + self.clf_b
        self.test_auroc.update(torch.sigmoid(neg_logits), torch.zeros_like(neg_similarity))
        self.test_auroc.update(torch.sigmoid(pos_logits), torch.ones_like(pos_similarity))


        similarity_compare = -pos_similarity[None, :].expand(B, B) \
                        + neg_similarity[:, None].expand(B, B)
        success_rate = (similarity_compare < 0).float().mean()

        self.test_auroc.update(torch.sigmoid(neg_logits), torch.zeros_like(neg_similarity))
        self.test_auroc.update(torch.sigmoid(pos_logits), torch.ones_like(pos_similarity))
        self.test_success_rate.update(success_rate.item())

    def on_train_epoch_end(self):
        sim_neg = self.train_sim_neg.compute()
        print(f"train_sim_negative: {sim_neg:.3f} \n")
        self.train_sim_neg.reset()
        
        sim_pos = self.train_sim_pos.compute()
        print(f"train_sim_positive: {sim_pos:.3f} \n")
        self.train_sim_pos.reset()

        train_auroc = self.train_auroc.compute()
        print(f"train_auroc: {train_auroc:.3f} \n")
        self.train_auroc.reset()
        self.log('train_auroc', train_auroc, sync_dist=True)

        train_acc = self.train_acc.compute()
        print(f"train_acc: {train_acc:.3f} \n")
        self.train_acc.reset()
        self.log('train_acc', train_acc, sync_dist=True)

        train_loss_pos = self.train_loss_pos.compute()
        print(f"train_loss_pos: {train_loss_pos:.3f} \n")
        self.train_loss_pos.reset()
        self.log('train_loss_pos', train_loss_pos, sync_dist=True)

        train_loss_neg = self.train_loss_neg.compute()
        print(f"train_loss_neg: {train_loss_neg:.3f} \n")
        self.train_loss_neg.reset()
        self.log('train_loss_neg', train_loss_neg, sync_dist=True)

        # train_loss = self.train_loss.compute()
        # print(f"train_loss: {train_loss:.3f} \n")
        # self.train_loss.reset()
        # self.log('train_loss', train_loss, sync_dist=True)

        pass

    def on_validation_epoch_end(self):

        sim_neg = self.valid_sim_neg.compute()
        print(f"valid_sim_negative: {sim_neg:.3f} \n")
        self.valid_sim_neg.reset()
        
        sim_pos = self.valid_sim_pos.compute()
        print(f"valid_sim_positive: {sim_pos:.3f} \n")
        self.valid_sim_pos.reset()

        valid_auroc = self.valid_auroc.compute()
        print(f"valid_auroc: {valid_auroc:.3f} \n")
        self.valid_auroc.reset()
        self.log('valid_auroc', valid_auroc, sync_dist=True)

        valid_acc = self.valid_acc.compute()
        print(f"valid_acc: {valid_acc:.3f} \n")
        self.valid_acc.reset()
        self.log('valid_acc', valid_acc, sync_dist=True)

        valid_loss_pos = self.valid_loss_pos.compute()
        print(f"valid_loss_pos: {valid_loss_pos:.3f} \n")
        self.valid_loss_pos.reset()
        self.log('valid_loss_pos', valid_loss_pos, sync_dist=True)

        valid_loss_neg = self.valid_loss_neg.compute()
        print(f"valid_loss_neg: {valid_loss_neg:.3f} \n")
        self.valid_loss_neg.reset()
        self.log('valid_loss_neg', valid_loss_neg, sync_dist=True)

        self.log("valid_loss", (valid_loss_neg + valid_loss_pos) / 2, sync_dist=True)

        # valid_loss = self.valid_loss.compute()
        # print(f"valid_loss: {valid_loss:.3f} \n")
        # self.valid_loss.reset()
        # self.log('valid_loss', valid_loss, sync_dist=True)

        pass

    def on_test_epoch_end(self):

        sim_neg = self.test_sim_neg.compute()
        print(f"test_sim_negative: {sim_neg:.3f} \n")
        self.test_sim_neg.reset()
        
        sim_pos = self.test_sim_pos.compute()
        print(f"test_sim_positive: {sim_pos:.3f} \n")
        self.test_sim_pos.reset()

        success_rate = self.test_success_rate.compute()
        print(f"success_rate: {success_rate:.3f} \n")
        self.test_success_rate.reset()

        val_auroc = self.test_auroc.compute()
        print(f"test_auroc: {val_auroc:.3f} \n")
        self.test_auroc.reset()

        self.log('test_auroc', val_auroc, sync_dist=True)
        pass

    def configure_optimizers(self):
        # exclude the params of classifier
        # parameters = set(self.parameters()) - {self.clf_a} - {self.clf_b}
        # optimizer = optim.Adam([
        #         {'params': list(parameters)},
        #         {'params': self.clf_a, 'lr': 1e-3},
        #         {'params': self.clf_b, 'lr': 1e-3},
        #     ], lr=self.lr, weight_decay=0.0)
        
        if isinstance(self.preproc, DnCNN):
            # set lr=1e-4 for DnCNN 
            parameters = set(self.parameters()) - set(self.preproc.parameters())
            optimizer = optim.Adam([
                    {'params': list(parameters)},
                    {'params': self.preproc.parameters(), 'lr': 1e-4},
                ], lr=self.lr, weight_decay=0.0)
        else:
            optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.0)
        
        # optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.0)  # don't add weight decay, otherwise the training doesn't converge
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99, verbose=True)
        return [optimizer], [scheduler]

    
    
