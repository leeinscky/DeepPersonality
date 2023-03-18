"""
code modified from https://github.com/YuanGongND/ssast

"""
import torch.nn as nn
import torch
import sys
sys.path.append("/data/sls/scratch/yuangong/aed-trans/src/models/")
sys.path.append("/data/sls/scratch/yuangong/aed-trans/src/")
from timm.models.layers import trunc_normal_
import timm
import numpy as np
from timm.models.layers import to_2tuple
from random import randrange
from matplotlib import pyplot as plt
import random
from .build import NETWORK_REGISTRY

# override the timm package to relax the input shape constraint.
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

class ASTModel(nn.Module):
    def __init__(self, label_dim=527,
                 fshape=128, tshape=2, fstride=128, tstride=2,
                 input_fdim=128, input_tdim=1024, model_size='base',
                 pretrain_stage=True, load_pretrained_mdl_path=None):

        super(ASTModel, self).__init__()
        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'

        # override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        # pretrain the AST models
        if pretrain_stage == True:
            if load_pretrained_mdl_path != None:
                raise ValueError('Setting load_pretrained_mdl_path at pretraining stage is useless, pretraining is always from scratch, please change it to None.')
            if fstride != fshape or tstride != tshape:
                raise ValueError('fstride != fshape or tstride != tshape, they must be same at the pretraining stage, patch split overlapping is not supported.')

            # if AudioSet pretraining is not used (but ImageNet pretraining may still apply)
            if model_size == 'tiny':
                print('Loading DeiT-tiny model...')
                self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=False)
                self.heads, self.depth = 3, 12
                self.cls_token_num = 2
            elif model_size == 'small':
                print('Loading DeiT-small model...')
                self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=False)
                self.heads, self.depth = 6, 12
                self.cls_token_num = 2
            elif model_size == 'base':
                print('Loading DeiT-base model...')
                self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=False)
                self.heads, self.depth = 12, 12
                self.cls_token_num = 2
            elif model_size == 'base_nokd':
                print('Loading DeiT-base model without knowledge distillation...')
                self.v = timm.create_model('vit_deit_base_patch16_384', pretrained=False)
                self.heads, self.depth = 12, 12
                self.cls_token_num = 1
            else:
                raise Exception('Model size must be one of tiny, small, base, base_nokd')

            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches ** 0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]

            # SSL Pretraining Code
            self.softmax = nn.Softmax(dim=-1)
            self.lsoftmax = nn.LogSoftmax(dim=-1)
            self.fshape, self.tshape = fshape, tshape
            self.fstride, self.tstride = fstride, tstride
            self.input_fdim, self.input_tdim = input_fdim, input_tdim
            # this is a trick to make state_dict to track pretraining input_fdim and input_tdim and save them by using torch.save
            self.p_input_fdim, self.p_input_tdim = nn.Parameter(torch.tensor(input_fdim), requires_grad=False), nn.Parameter(torch.tensor(input_tdim), requires_grad=False)

            # masked patch classification (discriminative objective) layer
            # we use two layers for pretext task, but using a single layer has similar performance.
            # we map the output of transformer (768-dim for base models) to 256-dim patch input space, and then dot product with flattened patch input (also 256-dim) to calculate loss.
            # alternatively, you can map the output of transformer to 768-dim patch embedding space, and dot product with patch embedding. Performance-wise they are similar, but map to 256 space is more efficient.
            self.cpredlayer = nn.Sequential(nn.Linear(self.original_embedding_dim, self.original_embedding_dim), nn.ReLU(), nn.Linear(self.original_embedding_dim, 256))
            # masked patch reconstruction (generative objective) layer
            self.gpredlayer = nn.Sequential(nn.Linear(self.original_embedding_dim, self.original_embedding_dim), nn.ReLU(), nn.Linear(self.original_embedding_dim, 256))
            self.unfold = torch.nn.Unfold(kernel_size=(fshape, tshape), stride=(fstride, tstride))

            # we use learnable mask embedding (follow the BEIT paper), but using a fixed mask embedding (e.g., 0) leads to same performance.
            self.mask_embed = nn.Parameter(torch.zeros([1, 1, self.original_embedding_dim]))
            self.mask_embed = torch.nn.init.xavier_normal_(self.mask_embed)

            # get the intermediate shape
            print('[ASTModel ****重要] 参数fstride:', fstride, 'tstride:', tstride, 'input_fdim:', input_fdim, 'input_tdim:', input_tdim, 'fshape:', fshape, 'tshape:', tshape)
            self.p_f_dim, self.p_t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim, fshape, tshape)
            num_patches = self.p_f_dim * self.p_t_dim
            self.num_patches = num_patches
            self.v.patch_embed.num_patches = num_patches
            print('[ASTModel] pretraining patch split stride: frequency={:d}, time={:d}'.format(fstride, tstride))
            print('[ASTModel] pretraining patch shape: frequency={:d}, time={:d}'.format(fshape, tshape))
            print('[ASTModel] pretraining patch array dimension: frequency={:d}, time={:d}'.format(self.p_f_dim, self.p_t_dim))
            print('[ASTModel] pretraining number of patches={:d}'.format(num_patches))

            # the linear patch projection layer, use 1 channel for spectrogram rather than the original 3 channels for RGB images.
            new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(fshape, tshape), stride=(fstride, tstride))
            self.v.patch_embed.proj = new_proj

            # use trainable positional embedding
            new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + self.cls_token_num, self.original_embedding_dim))
            self.v.pos_embed = new_pos_embed
            trunc_normal_(self.v.pos_embed, std=.02)

        # use a pretrained models for finetuning
        elif pretrain_stage == False:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if load_pretrained_mdl_path == None:
                raise ValueError('Please set load_pretrained_mdl_path to load a pretrained models.')
            sd = torch.load(load_pretrained_mdl_path, map_location=device)
            # get the fshape and tshape, input_fdim and input_tdim in the pretraining stage
            try:
                p_fshape, p_tshape = sd['module.v.patch_embed.proj.weight'].shape[2], sd['module.v.patch_embed.proj.weight'].shape[3]
                p_input_fdim, p_input_tdim = sd['module.p_input_fdim'].item(), sd['module.p_input_tdim'].item()
            except:
                raise  ValueError('The model loaded is not from a torch.nn.Dataparallel object. Wrap it with torch.nn.Dataparallel and try again.')

            print('now load a SSL pretrained models from ' + load_pretrained_mdl_path)
            # during pretraining, fstride=fshape and tstride=tshape because no patch overlapping is used
            # here, input_fdim and input_tdim should be that used in pretraining, not that in the fine-tuning.
            # we need to know input_fdim and input_tdim to do positional embedding cut/interpolation.
            # generally it should be better to use same input_fdim during pretraining and finetuning, but input_tdim can be safely different
            audio_model = ASTModel(fstride=p_fshape, tstride=p_tshape, fshape=p_fshape, tshape=p_tshape,
                                   input_fdim=p_input_fdim, input_tdim=p_input_tdim, pretrain_stage=True, model_size=model_size)
            audio_model = torch.nn.DataParallel(audio_model)
            audio_model.load_state_dict(sd, strict=False)

            self.v = audio_model.module.v
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.cls_token_num = audio_model.module.cls_token_num

            # mlp head for fine-tuning
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim),
                                          nn.Linear(self.original_embedding_dim, label_dim))

            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim, fshape, tshape)
            # patch array dimension during pretraining
            p_f_dim, p_t_dim = audio_model.module.p_f_dim, audio_model.module.p_t_dim
            num_patches = f_dim * t_dim
            p_num_patches = p_f_dim * p_t_dim
            self.v.patch_embed.num_patches = num_patches
            print('fine-tuning patch split stride: frequncey={:d}, time={:d}'.format(fstride, tstride))
            print('fine-tuning number of patches={:d}'.format(num_patches))

            # patch shape should be same for pretraining and fine-tuning
            if fshape != p_fshape or tshape != p_tshape:
                raise ValueError('The patch shape of pretraining and fine-tuning is not consistant, pretraining: f={:d}, t={:d}, finetuning: f={:d}, t={:d}'.format(p_fshape, p_tshape, fshape, tshape))

            # patch split stride generally should be different for pretraining and fine-tuning, as patch split overlapping is only used in finetuning
            # during pretraining, p_fshape = p_fstride and p_tshape = p_tstride
            if fstride != p_fshape or tstride != p_tshape:
                # initialize a new patch embedding layer with desired new stride.
                new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(fshape, tshape), stride=(fstride, tstride))
                # but the weights of patch embedding layer is still got from the pretrained models
                new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
                new_proj.bias = self.v.patch_embed.proj.bias
                self.v.patch_embed.proj = new_proj

            new_pos_embed = self.v.pos_embed[:, self.cls_token_num:, :].detach().reshape(1, p_num_patches, self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, p_f_dim, p_t_dim)
            # cut or interpolate the positional embedding
            if t_dim < p_t_dim:
                new_pos_embed = new_pos_embed[:, :, :, int(p_t_dim/2) - int(t_dim / 2): int(p_t_dim/2) - int(t_dim / 2) + t_dim]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(8, t_dim), mode='bilinear')
            if f_dim < p_f_dim:
                new_pos_embed = new_pos_embed[:, :, int(p_f_dim/2) - int(f_dim / 2): int(p_f_dim/2) - int(f_dim / 2) + t_dim, :]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')

            new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :self.cls_token_num, :].detach(), new_pos_embed], dim=1))

    # get the shape of intermediate representation.
    def get_shape(self, fstride, tstride, input_fdim, input_tdim, fshape, tshape):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(fshape, tshape), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    # generate mask for 16*16 patch
    def gen_maskid_patch(self, sequence_len=512, mask_size=100, cluster=3):
        mask_id = []

        # randomize clutering factor in [3,6)
        cur_clus = randrange(cluster) + 3
        # print('[ASTModel - gen_maskid_patch] sequence_len:', sequence_len, 'mask_size:', mask_size, 'cluster:', cluster, ', cur_clus:', cur_clus)
        # sequence_len: 128 mask_size: 400 cluster: 3 , cur_clus: 4
        
        while len(list(set(mask_id))) <= mask_size:
            # print('[ASTModel - gen_maskid_patch] len(mask_id):', len(list(set(mask_id))) , 'mask_size:', mask_size)
            start_id = randrange(sequence_len)

            # this improves the efficiency, but might change the pretrained model
            # while start_id in mask_id:
            #     start_id = randrange(sequence_len)

            cur_mask = []
            for i in range(0, cur_clus):
                # print('[ASTModel - gen_maskid_patch] i:', i, ', cur_clus:', cur_clus)
                for j in range(0, cur_clus):
                    # print('[ASTModel - gen_maskid_patch] j:', j)
                    mask_cand = start_id + self.p_t_dim * i + j
                    # print('[ASTModel - gen_maskid_patch] mask_cand:', mask_cand, 'sequence_len:', sequence_len)
                    if mask_cand > 0 and mask_cand < sequence_len:
                        # print('[ASTModel - gen_maskid_patch] append')
                        cur_mask.append(mask_cand)
                        # print('[ASTModel - gen_maskid_patch] len(cur_mask', len(cur_mask))
            mask_id = mask_id + cur_mask
            # print('[ASTModel - gen_maskid_patch] after for loop, mask_id len:', len(list(set(mask_id))) , 'mask_size:', mask_size)
        mask_id = list(set(mask_id))[:mask_size]
        # print('[ASTModel - gen_maskid_patch] mask_id.shape:', torch.tensor(mask_id).shape)
        return torch.tensor(mask_id)

    # using cluster for frame masking hurts the performance, so just use the naive random sampling
    def gen_maskid_frame(self, sequence_len=512, mask_size=100):
        mask_id = random.sample(range(0, sequence_len), mask_size)
        return torch.tensor(mask_id)

    def finetuningavgtok(self, x):
        # print('[ASTModel - finetuningavgtok] input x.shape:', x.shape)  # [8, 1, 256, 1598] (B, C, F, T)
        B = x.shape[0]
        x = self.v.patch_embed(x)
        # print('[ASTModel - finetuningavgtok] after patch_embed, x.shape:', x.shape)
        if self.cls_token_num == 2:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            dist_token = self.v.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
            # print('[ASTModel - finetuningavgtok] after cls_token, x.shape:', x.shape, 'cls_tokens.shape:', cls_tokens.shape, 'dist_token.shape:', dist_token.shape)
        else:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            # print('[ASTModel - finetuningavgtok] after cls_token, x.shape:', x.shape, 'cls_tokens.shape:', cls_tokens.shape)
        x = x + self.v.pos_embed
        # print('[ASTModel - finetuningavgtok] after pos_embed, x.shape:', x.shape)
        x = self.v.pos_drop(x)
        # print('[ASTModel - finetuningavgtok] after pos_drop, x.shape:', x.shape)

        for blk_id, blk in enumerate(self.v.blocks):
            x = blk(x)
        # print('[ASTModel - finetuningavgtok] after blocks, x.shape:', x.shape)
        x = self.v.norm(x)
        # print('[ASTModel - finetuningavgtok] after norm, x.shape:', x.shape)

        # average output of all tokens except cls token(s)
        x = torch.mean(x[:, self.cls_token_num:, :], dim=1)
        # print('[ASTModel - finetuningavgtok] after mean, x.shape:', x.shape)
        x = self.mlp_head(x)
        # print('[ASTModel - finetuningavgtok] after mlp_head, final return x.shape:', x.shape)
        
        # add sigmoid to the output
        x = torch.sigmoid(x)
        return x

    def finetuningcls(self, x):
        print('[ASTModel - finetuningcls] input x.shape:', x.shape) # e.g. [8, 1, 256, 1598] (B, C, H, W)
        B = x.shape[0]
        x = self.v.patch_embed(x)
        print('[ASTModel - finetuningcls] after patch_embed, x.shape:', x.shape)
        if self.cls_token_num == 2:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            dist_token = self.v.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
            print('[ASTModel - finetuningcls] after cat, x.shape:', x.shape, 'cls_tokens.shape:', cls_tokens.shape, 'dist_token.shape:', dist_token.shape)
        else:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            print('[ASTModel - finetuningcls] after cat, x.shape:', x.shape, 'cls_tokens.shape:', cls_tokens.shape)
        x = x + self.v.pos_embed
        print('[ASTModel - finetuningcls] after pos_embed, x.shape:', x.shape)
        x = self.v.pos_drop(x)
        print('[ASTModel - finetuningcls] after pos_drop, x.shape:', x.shape)

        for blk_id, blk in enumerate(self.v.blocks):
            x = blk(x)
        print('[ASTModel - finetuningcls] after blocks, x.shape:', x.shape)
        x = self.v.norm(x)
        print('[ASTModel - finetuningcls] after norm, x.shape:', x.shape)

        # if models has two cls tokens (DEIT), average as the clip-level representation
        if self.cls_token_num == 2:
            x = (x[:, 0] + x[:, 1]) / 2
            print('[ASTModel - finetuningcls] after average, x.shape:', x.shape)
        else:
            x = x[:, 0] # take the first cls token
            print('[ASTModel - finetuningcls] after x[:, 0], x.shape:', x.shape)
        x = self.mlp_head(x)
        print('[ASTModel - finetuningcls] after mlp_head, final return x.shape:', x.shape)
        
        # add sigmoid to the output
        x = torch.sigmoid(x)
        
        return x

    # masked patch pretraining with discriminative objective
    def mpc(self, x, mask_patch, cluster, show_mask=False):
        # print('[ASTModel - mpc] x.shape: ', x.shape, ', mask_patch: ', mask_patch, ', cluster: ', cluster)
        input = self.unfold(x).transpose(1, 2)
        B = x.shape[0]
        # x in shape (batch_size, sequence_len, embedding dim)
        x = self.v.patch_embed(x)
        # print('[ASTModel - mpc] after patch_embed, x.shape: ', x.shape)

        # encode the patch
        # size 12(batch_size) * 100(#mask_patch) * 768(hidden_dim), prepare to save the true values of masked samples
        encode_samples = torch.empty((B, mask_patch, 256), device=x.device, requires_grad=False).float()
        # size 12(batch_size) * 100(#mask_patch), index of masked patches
        mask_index = torch.empty((B, mask_patch), device=x.device, requires_grad=False).long()
        # size 12(batch_size) * 512(sequence_len) * 768(hidden_dim)
        mask_dense = torch.ones([x.shape[0], x.shape[1], x.shape[2]], device=x.device)
        # print('[ASTModel - mpc] encode_samples.shape: ', encode_samples.shape, ', mask_index.shape: ', mask_index.shape, ', mask_dense.shape: ', mask_dense.shape)

        # for each audio clip in the batch
        for i in range(B):
            # randomly generate #mask_patch mask indexes without duplicate
            if cluster == True:
                # use this if you are masking e.g. 16*16 patches
                # print('[ASTModel - mpc] i:', i, ', self.num_patches:', self.num_patches, ', mask_patch:', mask_patch) # self.num_patches:  128 , mask_patch:  400
                mask_index[i] = self.gen_maskid_patch(self.num_patches, mask_patch)
                # print('[ASTModel - mpc] i:', i, ', mask_index[i]: ', mask_index[i])
            else:
                # use this if you are masking frame, i.e., 128*2 patches
                mask_index[i] = self.gen_maskid_frame(self.num_patches, mask_patch)
            # copy the masked embeddings, note gradients are stopped in this path
            encode_samples[i] = input[i, mask_index[i], :].clone().detach()
            # mask the encode samples with 0
            mask_dense[i, mask_index[i], :] = 0

        # follow BEIT paper, mask with learnable masking embedding, but no performance diff observed compared with masking with 0s.
        mask_tokens = self.mask_embed.expand(B, x.shape[1], -1)
        # print('[ASTModel - mpc] mask_tokens.shape: ', mask_tokens.shape)

        # mask the patch
        x = x * mask_dense + (1-mask_dense) * mask_tokens
        # print('[ASTModel - mpc] after masking, x.shape: ', x.shape)

        # pass through the Transformer layers
        cls_tokens = self.v.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        # print('[ASTModel - mpc] after concat, cls_tokens.shape: ', cls_tokens.shape, ', dist_token.shape: ', dist_token.shape, ', x.shape: ', x.shape)
        x = x + self.v.pos_embed
        # print('[ASTModel - mpc] after pos_embed, x.shape: ', x.shape)
        x = self.v.pos_drop(x)
        # print('[ASTModel - mpc] after pos_drop, x.shape: ', x.shape)
        for blk in self.v.blocks:
            x = blk(x)
        # print('[ASTModel - mpc] after blocks, x.shape: ', x.shape)
        x = self.v.norm(x)
        # print('[ASTModel - mpc] after norm, x.shape: ', x.shape) # .shape:  torch.Size([4, 1586, 192])

        # prediction of the masked patch
        pred = torch.empty((B, mask_patch, 256), device=x.device).float()  # e.g. size 12*100*768
        
        for i in range(B):
            #  +2 for indexes because skipping the cls and dis token
            # we map the output of transformer (768-dim for base models) to 256-dim patch input space, and then dot product with flattened patch input (also 256-dim) to calculate loss.
            # alternatively, you can map the output of transformer to 768-dim patch embedding space, and dot product with patch embedding. Performance-wise they are similar, but map to 256 space is more efficient.
            pred[i] = self.cpredlayer(x[i, mask_index[i] + self.cls_token_num, :])
        # print('[ASTModel - mpc] pred.shape: ', pred.shape) # pred.shape:  torch.Size([batch_size, 400, 256])

        # calculate the NCE loss
        nce = torch.tensor(0.0).to(x.device)
        correct = torch.tensor(0.0).to(x.device)
        for i in np.arange(0, B):
            # negative samples are from the same batch
            # 8/12/2022: has a difference with equation (1) in the ssast paper but (likely) performance-wise similar, see https://github.com/YuanGongND/ssast/issues/13
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))  # e.g. size 100*100
            correct += torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, mask_patch, device=x.device)))  # correct is a tensor
            nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
        
        acc = 1. * correct / (B * mask_patch)
        nce = nce / (-1. * B * mask_patch)
        # print('[ASTModel - mpc] correct: ', correct, ', B: ', B, ', mask_patch: ', mask_patch, ', acc: ', acc, ', nce: ', nce)

        # visualize the masked area, for probing test only, set show_mask = False for any training/inference.
        if show_mask == False:
            # print('[ASTModel - mpc] show_mask: ', show_mask, ', return acc: ', acc, ', nce: ', nce)
            return acc, nce
        else:
            if B > 1:
                raise Exception('Currently only support single spectrogram probing test.')

            self.mask_correct = torch.nn.Parameter(torch.arange(0, mask_patch), requires_grad=False)

            pred = input.clone()  # [B, 512, 256]
            masked = input.clone()

            for i in range(B):
                result = [float(t) * 99 for t in torch.eq(torch.argmax(self.softmax(total), dim=0), self.mask_correct)]
                pred[i, mask_index[i], :] = torch.tensor(result).reshape(mask_patch, 1).expand(mask_patch, 256)
                masked[i, mask_index[i], :] = 99.0

            # print(total)
            # print(self.softmax(total))
            # print(torch.argmax(self.softmax(total), dim=0))
            # print(self.mask_correct)
            # print(torch.eq(torch.argmax(self.softmax(total), dim=0), self.mask_correct))
            # print([float(t)*99 for t in torch.eq(torch.argmax(self.softmax(total), dim=0), self.mask_correct)])

            fold = torch.nn.Fold(output_size=([self.input_fdim, self.input_tdim]), kernel_size=(self.fshape, self.tshape), stride=(self.fstride, self.tstride))
            pred = fold(pred.transpose(1, 2))
            masked = fold(masked.transpose(1, 2))

            # print('[ASTModel] pred.shape: ', pred.shape, 'masked.shape: ', masked.shape)
            return pred, masked

    # # masked patch pretraining with generative objective
    def mpg(self, input, mask_patch, cluster):
        B = input.shape[0]
        x = self.v.patch_embed(input)
        input = self.unfold(input).transpose(1, 2)

        # size 12(batch_size) * 100(#mask_patch), index of masked patches
        mask_index = torch.empty((B, mask_patch), device=x.device, requires_grad=False).long()
        # size 12(batch_size) * 512(sequence_len) * 768(hidden_dim)
        mask_dense = torch.ones([x.shape[0], x.shape[1], x.shape[2]], device=x.device)
        for i in range(B):
            # randomly generate #mask_patch mask indexes without duplicate
            if cluster == True:
                # use this if you are masking e.g. 16*16 patches
                mask_index[i] = self.gen_maskid_patch(self.num_patches, mask_patch)
            else:
                # use this if you are masking frame, i.e., 128*2 patches
                mask_index[i] = self.gen_maskid_frame(self.num_patches, mask_patch)
            mask_dense[i, mask_index[i], :] = 0

        mask_tokens = self.mask_embed.expand(B, x.shape[1], -1)

        # follow BEIT paper, mask with learnable masking embedding, but no performance diff observed compared with masking with 0s.
        x = x * mask_dense + (1-mask_dense) * mask_tokens

        # go through the Transformer layers
        cls_tokens = self.v.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)

        pred = torch.empty((B, mask_patch, self.fshape * self.tshape), device=x.device).float()  # e.g. size 12*100*256
        target = torch.empty((B, mask_patch, self.fshape * self.tshape), device=x.device).float() # e.g. size 12*100*256

        for i in range(B):
            #  +2 for indexes because cls and dis token
            pred[i] = self.gpredlayer(x[i, mask_index[i] + self.cls_token_num, :])
            target[i] = input[i, mask_index[i], :]

        # calculate the MSE loss
        mse = torch.mean((pred - target) ** 2)

        return mse

    def forward(self, x, task='pretrain_mpc', cluster=True, mask_patch=400):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        
        # print('[ASTModel] input x.shape: ', x.shape, ', task: ', task, ', cluster: ', cluster, ', mask_patch: ', mask_patch)
        x = x.unsqueeze(1) # (batch_size, 1, time_frame_num, frequency_bins), e.g., (12, 1, 1024, 128)
        # print('[ASTModel] input x.unsqueeze(1).shape: ', x.shape)
        x = x.transpose(2, 3) # (batch_size, 1, frequency_bins, time_frame_num), e.g., (12, 1, 128, 1024)
        # print('[ASTModel] input x.transpose(2, 3).shape: ', x.shape)

        # finetuning (ft), use the mean of all token (patch) output as clip-level representation.
        # this is default for SSAST fine-tuning as during pretraining, supervision signal is given to each token, not the [cls] token
        if task == 'ft_avgtok':
            return self.finetuningavgtok(x)
        # alternatively, use the [cls] token output as clip-level representation.
        elif task == 'ft_cls':
            return self.finetuningcls(x)
        # pretraining, masked patch classification (discriminative objective)
        elif task == 'pretrain_mpc':
            return self.mpc(x, mask_patch=mask_patch, cluster=cluster)
        # pretraining, masked patch reconstruction (generative objective)
        elif task == 'pretrain_mpg':
            return self.mpg(x, mask_patch=mask_patch, cluster=cluster)
        elif task == 'visualize_mask':
            return self.mpc(x, mask_patch=mask_patch, cluster=cluster, show_mask=True)
        else:
            raise Exception('Task unrecognized.')



@NETWORK_REGISTRY.register()
def ssast_udiva(cfg):
    num_mel_bins = 256 # 128(default) * 2(2个视频) = 256
    # target_length = 1598 # fbank的输出长度 
    target_length = (cfg.DATA.SAMPLE_SIZE/5 * 100) - 2  # fbank的输出长度 #根据对应关系找出来的规律 sample_size=16秒: fbank.shape: torch.Size([1598, 128]) # sample_size=32秒: fbank.shape: torch.Size([3198, 128]) # sample_size=48秒: fbank.shape: torch.Size([4798, 128]) # sample_size=64秒: fbank.shape: torch.Size([6398, 128])
    # print('num_mel_bins: ', num_mel_bins, ', target_length: ', target_length)
    if cfg.TRAIN.PRE_TRAINED_MODEL is None: # 如果没有预训练模型，即进行预训练
        print('Pretrain AST model...')
        is_pretrain = True # 预训练状态为True
        pretrained_model_path = None # 预训练模型路径为None
    else:
        print('Finetune AST model...')
        is_pretrain = False # 预训练状态为False
        pretrained_model_path = cfg.TRAIN.PRE_TRAINED_MODEL # 预训练模型路径为cfg.TRAIN.PRE_TRAINED_MODEL
        
    model = ASTModel(
                 label_dim=cfg.MODEL.NUM_CLASS, fshape=16, tshape=16, fstride=16, tstride=16,
                 input_fdim=num_mel_bins, input_tdim=int(target_length),
                 model_size='tiny', pretrain_stage=is_pretrain, load_pretrained_mdl_path=pretrained_model_path)
    """ 
    class ASTModel(nn.Module):
        def __init__(self, label_dim=527,
                    fshape=128, tshape=2, fstride=128, tstride=2,
                    input_fdim=128, input_tdim=1024, model_size='base',
                    pretrain_stage=True, load_pretrained_mdl_path=None):
    """
    return model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))


if __name__ == '__main__':
    # this is an example of how to use the SSAST model

    # pretraining stage
    # suppose you have an unlabled dataset with avg length of 1024 frames (i.e., 10.24s)
    input_tdim = 1024
    # create a 16*16 patch based AST model for pretraining.
    # note, we don't use patch split overlap in pretraining, so fstride=fshape and tstride=tshape
    ast_mdl = ASTModel(
                 fshape=16, tshape=16, fstride=16, tstride=16,
                 input_fdim=128, input_tdim=input_tdim, model_size='base',
                 pretrain_stage=True)
    # # alternatively, create a frame based AST model
    # ast_mdl = ASTModel(
    #              fshape=128, tshape=2, fstride=128, tstride=2,
    #              input_fdim=128, input_tdim=input_tdim, model_size='base',
    #              pretrain=True)

    # do pretraining, see src/traintest_mask.py for our full pretraining code
    # input in shape [batch_size, input_tdim, input_fdim]
    test_input = torch.zeros([10, input_tdim, 128])
    # mask 100 patches for both discriminative and generative loss
    acc, nce_loss = ast_mdl(test_input, task='pretrain_mpc', mask_patch=100)
    mse_loss = ast_mdl(test_input, task='pretrain_mpg', mask_patch=100)
    loss = nce_loss + 10 * mse_loss
    # do back propagate and update the model, etc

    # after pretraining, save the pretrained model.
    # the code is designed for Dataparallel model
    ast_mdl = torch.nn.DataParallel(ast_mdl)
    torch.save(ast_mdl.state_dict(), './test_mdl.pth')

    # fine-tuning stage
    # now you have a labeled dataset you want to finetune AST on
    # suppose the avg length is 100 frames (1s) and there are 35 classes
    # the fshape and tshape must be same in pretraining and finetuning
    # but fstride and tstride can be different in pretraining and finetuning
    # using smaller strides improves the performance but also increase the computational overhead
    # set pretrain_stage as False since now is in the finetuning stage
    # provide the path of the pretrained model you want to load
    input_tdim = 100  # fine-tuning data length can be different with pretraining data length
    ast_mdl = ASTModel(label_dim=35,
                 fshape=16, tshape=16, fstride=10, tstride=10,
                 input_fdim=128, input_tdim=input_tdim, model_size='base',
                 pretrain_stage=False, load_pretrained_mdl_path='./test_mdl.pth')
    # # alternatively, use a frame based AST model
    # ast_mdl = ASTModel(label_dim=35,
    #              fshape=128, tshape=2, fstride=128, tstride=1,
    #              input_fdim=128, input_tdim=input_tdim, model_size='base',
    #              pretrain_stage=False, load_pretrained_mdl_path='./test_mdl.pth')

    # do finetuning, see src/traintest.py for our finetuning code
    test_input = torch.zeros([10, input_tdim, 128])
    prediction = ast_mdl(test_input, task='ft_avgtok')
    # output should in shape [batch_size, label_dim]
    print(prediction.shape)
    # calculate the loss, do back propagate, etc

    # # (optional) do some probe test
    # test_input = torch.zeros([1, input_tdim, 128]).to(device)
    # acc, nce = ast_mdl(test_input, task='pretrain_mpc', mask_patch=100)
    # # you can visualize the mask
    # pred, masked = ast_mdl(test_input, task='visualize_mask', mask_patch=100)
    # plt.imshow(masked[0,0])
    # plt.show()
