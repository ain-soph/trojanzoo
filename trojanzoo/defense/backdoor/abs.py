# -*- coding: utf-8 -*-

from ..defense_backdoor import Defense_Backdoor
from trojanzoo.utils import to_tensor, repeat_to_batch
from trojanzoo.utils.model import TVLoss

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os

from typing import Dict

from trojanzoo.utils import Config
env = Config.env

tv_loss = TVLoss()


class ABS(Defense_Backdoor):

    name = 'abs'

    def __init__(self, use_mask: bool = True, count_mask=True, seed_num: int = 50,
                 samp_k: int = 1, same_range: bool = False, n_samples: int = 5, samp_batch_size: int = 8,
                 max_troj_size: int = 16, re_mask_lr: float = 0.1, re_mask_weight: float = 500, re_iteration: int = 1000, **kwargs):
        super().__init__(**kwargs)
        self.use_mask: bool = use_mask
        self.count_mask: bool = count_mask
        self.seed_num: int = seed_num

        # -----------Neural Sampling------------- #
        self.samp_k: int = samp_k
        self.same_range: bool = same_range
        self.n_samples: int = n_samples
        self.samp_batch_size: int = samp_batch_size
        # self.top_n_neurons: int = 20

        # ----------------Remask----------------- #
        self.max_troj_size: int = max_troj_size
        self.re_mask_lr: float = re_mask_lr
        self.re_mask_weight: float = re_mask_weight
        self.re_iteration: int = re_iteration

        self.seed_data = self.load_seed_img()
        self.nc_mask = self.nc_filter_img()

    def detect(self):
        seed_data = self.load_seed_data()
        _input, _label = seed_data['input'], seed_data['label']
        length = len(_label) // 2
        train_xs = _input[:length]
        train_ys = _label[:length]
        test_xs = _input[length:]
        test_ys = _label[length:]
        all_ps = self.sample_neuron(train_xs, train_ys)
        neuron_dict = self.find_min_max(all_ps)
        results = self.re_mask(neuron_dict, train_xs)
        reasrs = []
        for result in results:
            reasr = self.test_mask(weights_file, test_xs, result)
            adv, rdelta, rmask, Troj_Label, RE_img, RE_mask, RE_delta, Troj_Layer, acc = result
            print(Troj_Layer)
            print('train acc: ', acc)
            print('test  acc: ', reasr)
            print()
            reasrs.append(reasr)
            if reasr > 80:
                adv, rdelta, rmask, Troj_Label, RE_img, RE_mask, RE_delta, Troj_Layer, acc = result
                for i in range(adv.shape[0]):
                    save_tensor_as_img(
                        RE_img[:-4] + ('_{0}.png').format(i), adv[i])
                np.save(RE_delta, to_numpy(rdelta))
            with open(RE_mask, 'wb') as (f):
                pickle.dump(rmask, f)

    # -----------------------Test Mask--------------------------------- #

    def stamp(self, n_img, delta, mask):
        mask0 = self.nc_filter_img(self.h, self.w, use_mask=self.use_mask)
        mask = mask * mask0
        r_img = n_img * (1 - mask) + delta * mask
        return r_img

    def test_mask(self, weights_file, test_xs, result):
        rimg, rdelta, rmask, tlabel = result[:4]
        self.model.load_pretrained_weights(weights_file)
        t_images = self.stamp(test_xs, rdelta, rmask)
        for i in range(len(t_images)):
            save_numpy_as_img(self.folder_path + '/{0}.png'.format(i), t_images[i])

        yt = int(tlabel) * torch.ones(len(t_images),
                                      dtype=torch.long, device=self.model.device)
        acc, _ = self.model.accuracy(self.model(t_images), yt)
        return acc

    # ---------------------------- Seed Data --------------------------- #
    def save_seed_data(self) -> Dict[str, np.ndarray]:
        torch.manual_seed(env['seed'])
        assert self.seed_num % self.model.num_classes == 0
        seed_class_num: int = self.seed_num // self.model.num_classes
        x, y = [], []
        for _class in range(self.model.num_classes):
            loader = self.dataset.get_dataloader(mode='train', batch_size=seed_class_num,
                                                 shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
            _input, _label = next(iter(loader))
            x.append(_input)
            y.append(_label)
        x = torch.stack(x).flatten(end_dim=1).numpy()
        y = torch.stack(y).flatten(end_dim=1).numpy()
        seed_data = {'input': x, 'label': y}
        seed_path = env['result_dir'] + '{0:s}/{1:s}_{2:d}.npy'.format(self.name, self.dataset.name, self.seed_num)
        np.save(seed_path, seed_data)
        print('seed data saved at: ', seed_path)
        return seed_data

    def load_seed_data(self) -> Dict[str, torch.Tensor]:
        seed_path = env['result_dir'] + '{0:s}/{1:s}_{2:d}.npy'.format(self.name, self.dataset.name, self.seed_num)
        seed_data: Dict[str, torch.Tensor] = {}
        seed_data = np.load(seed_path, allow_pickle=True).item() if os.path.exists(seed_path) \
            else self.save_seed_data()
        seed_data['input'] = to_tensor(seed_data['input'])
        seed_data['label'] = to_tensor(seed_data['label'], dtype=torch.long)
        return seed_data

    # -----------------------Neural Sample---------------------------- #

    def sample_neuron(self, _input: torch.Tensor, _label: torch.LongTensor):
        all_ps = {}
        batch_size = _input.shape[0]

        maxes = self.check_values(_input)

        layer_output = self.model.get_all_layer(_input)
        layer_name_list = self.model.get_layer_name()
        for layer in layer_name_list:
            if 'pool' in layer or layer in ['features', 'flatten', 'classifier', 'logits', 'output']:
                continue
            cur_layer_output = layer_output[layer]
            neuron_num = cur_layer_output.shape[-1]
            neuron_batch_number = neuron_num // self.samp_batch_size

            repeat_shape = [neuron_batch_number, self.n_samples]
            repeat_shape.extend([1] * len(cur_layer_output.shape))
            h_t: torch.Tensor = cur_layer_output.repeat(*repeat_shape)
            # (neuron_batch_number, self.n_samples, batch_size, C, H, W)

            vs = self.samp_k * torch.arange(self.n_samples)
            if not self.same_range:
                maxes = cur_layer_output.max()
                tr = float(maxes) / self.n_samples
                vs *= tr
            for neuron_batch_idx in range(neuron_batch_number):
                for neuron in range(self.samp_batch_size):
                    for i, v in enumerate(vs):
                        if len(cur_layer_output.shape) == 4:
                            h_t[neuron_batch_idx, i, :, :, :,
                                neuron + neuron_batch_idx * self.samp_batch_size] = v
                        elif len(cur_layer_output.shape) == 2:
                            h_t[neuron_batch_idx, i, :,
                                neuron + neuron_batch_idx * self.samp_batch_size] = v
                        else:
                            print('layer output shape: ',
                                  cur_layer_output.shape)
                            raise ValueError()
            f_h_t = h_t.flatten(end_dim=2)
            result = self.model.get_layer(f_h_t, layer_input=layer)
            result_shape = list(h_t.shape)[:3]
            result_shape.extend(list(result.shape)[1:])
            tps = result.view(result_shape)    # (neuron_batch_number, self.n_samples, batch_size, num_classes)
            for neuron_batch_idx in range(neuron_batch_number):
                for idx_img in range(batch_size):
                    ps_key = '{0}_{1}_{2}'.format(int(_label[idx_img]), layer, neuron_batch_idx)
                    # (n_samples, num_classes)
                    ps = tps[neuron_batch_idx, :, idx_img].transpose()
                    ps = ps.T  # (num_classes, n_samples)
                    all_ps[ps_key] = ps
        return all_ps

    @staticmethod
    def find_min_max(all_ps, cut_val=5, top_k=10):
        max_ps = {}
        max_vals = []
        n_classes = 0
        n_samples = 0
        for k in sorted(all_ps.keys()):
            all_ps[k] = all_ps[k][:, :cut_val]
            n_classes = all_ps[k].shape[0]
            n_samples = all_ps[k].shape[1]
            # if 'predictions_cifa10' in k or 'flatten' in k or 'dropout' in k:
            #     continue
            vs = []
            for l in range(10):
                vs.append(np.amax(all_ps[k][l][all_ps[k].shape[1] // 5:])
                          - np.amin(all_ps[k][l][:all_ps[k].shape[1] // 5]))

            ml = np.argsort(np.asarray(vs))[(-1)]
            sml = np.argsort(np.asarray(vs))[(-2)]
            val = vs[ml] - vs[sml]
            max_vals.append(val)
            max_ps[k] = (ml, val)

        neuron_ks = []
        imgs = []
        for k in sorted(max_ps.keys()):
            nk = ('_').join(k.split('_')[2:])
            neuron_ks.append(nk)
            imgs.append(('_').join(k.split('_')[:2]))
        neuron_ks = list(set(neuron_ks))
        imgs = list(set(imgs))
        min_ps = {}
        min_vals = []
        for k in neuron_ks:
            vs = []
            ls = []
            vdict = {}
            for img in sorted(imgs):
                nk = img + '_' + k
                l = max_ps[nk][0]
                v = max_ps[nk][1]
                vs.append(v)
                ls.append(l)
                if l not in vdict.keys():
                    vdict[l] = [v]
                else:
                    vdict[l].append(v)

            ml = max(set(ls), key=ls.count)
            tvs = []
            for img in sorted(imgs):
                nk = img + '_' + k
                l = max_ps[nk][0]
                v = max_ps[nk][1]
                tvs.append(v)

            fvs = []
            for img in sorted(imgs):
                img_l = int(img.split('_')[0])
                if img_l == ml:
                    continue
                nk = img + '_' + k
                l = max_ps[nk][0]
                v = max_ps[nk][1]
                if l != ml:
                    continue
                fvs.append(v)

            if len(fvs) == 0:
                for img in sorted(imgs):
                    img_l = int(img.split('_')[0])
                    if img_l == ml:
                        continue
                    nk = img + '_' + k
                    l = max_ps[nk][0]
                    v = max_ps[nk][1]
                    fvs.append(v)

            min_ps[k] = (
                l, ls.count(l), np.min(fvs), fvs)
            min_vals.append(np.min(fvs))

        keys = min_ps.keys()
        keys = []
        for k in min_ps.keys():
            if min_ps[k][1] >= n_samples - 2:
                keys.append(k)

        sorted_key = sorted(keys, key=lambda x: min_ps[x][2])
        neuron_dict = []
        maxval = min_ps[sorted_key[(-1)]][2]
        for i in range(min(len(sorted_key), top_k)):
            k = sorted_key[(-i - 1)]
            layer = k.split('_')[0]
            neuron = k.split('_')[(-1)]
            neuron_dict.append((layer, neuron, min_ps[k][0]))
        return neuron_dict

# -------------------------ReMask--------------------------------- #

    def re_mask_loss(self, neuron_dict, images, delta, mask):

        layers = self.model.get_layer_name()
        Troj_size = self.max_troj_size
        validated_results = []
        loss = torch.tensor([0.0], device=self.model.device)
        for task in neuron_dict:
            Troj_Layer, Troj_Neuron, Troj_Label = task
            Troj_Neuron = int(Troj_Neuron)
            Troj_next_Layer = layers[(layers.index(Troj_Layer))]
            Troj_next_Neuron = Troj_Neuron
            loss += self.abs_loss(images, delta, None, use_mask=mask,
                                  Troj_Layer=Troj_Layer, Troj_next_Layer=Troj_next_Layer,
                                  Troj_Neuron=Troj_Neuron, Troj_next_Neuron=Troj_next_Neuron, Troj_size=Troj_size)

        return loss

    def re_mask(self, neuron_dict, images, weights_file):
        layers = self.model.get_layer_name()
        Troj_size = self.max_troj_size
        validated_results = []
        for task in neuron_dict:
            Troj_Layer, Troj_Neuron, Troj_Label = task
            Troj_Neuron = int(Troj_Neuron)
            Troj_next_Layer = layers[(layers.index(Troj_Layer))]
            Troj_next_Neuron = Troj_Neuron
            optz_option = 0
            RE_img = (self.result_dir + 'imgs/{0}_model_{1}_{2}_{3}_{4}.png').format(weights_file.split(
                '/')[(-1)][:-3], Troj_Layer, Troj_Neuron, Troj_size, Troj_Label)
            RE_mask = (self.result_dir + 'masks/{0}_model_{1}_{2}_{3}_{4}.png').format(weights_file.split(
                '/')[(-1)][:-3], Troj_Layer, Troj_Neuron, Troj_size, Troj_Label)
            RE_delta = (self.result_dir + 'deltas/{0}_model_{1}_{2}_{3}_{4}.npy').format(weights_file.split(
                '/')[(-1)][:-3], Troj_Layer, Troj_Neuron, Troj_size, Troj_Label)
            acc, rimg, rdelta, rmask = self.reverse_engineer(optz_option, images, weights_file, Troj_Layer, Troj_Neuron,
                                                             Troj_next_Layer, Troj_next_Neuron, Troj_Label, RE_img, RE_delta, RE_mask, Troj_size)
            if acc >= 0:
                validated_results.append(
                    (rimg, rdelta, rmask, Troj_Label, RE_img, RE_mask, RE_delta, Troj_Layer, acc))
        return validated_results

    def reverse_engineer(self, optz_option, images, weights_file, Troj_Layer, Troj_Neuron, Troj_next_Layer, Troj_next_Neuron, Troj_Label, RE_img='./adv.png', RE_delta='./delta.pkl', RE_mask='./mask.pkl', Troj_size=64):

        if self.use_mask:
            mask = to_tensor(self.filter_img(self.h, self.w) * 4 - 2)
        else:
            mask = to_tensor(self.filter_img(self.h, self.w) * 8 - 4)
        delta = torch.randn(1, 3, self.h, self.w, device=self.model.device)
        delta.requires_grad = True
        mask.requires_grad = True

        self.model.load_pretrained_weights(weights_file)
        optimizer = optim.Adam([delta, mask] if self.use_mask else [delta],
                               lr=self.re_mask_lr)
        optimizer.zero_grad()

        # if optz_option == 0:
        #     delta = delta.view(1, self.h, self.w, 3)
        # elif optz_option == 1:
        #     delta = delta.view(-1, self.h, self.w, 3)

        self.model.eval()
        if optz_option == 0:
            for e in range(self.re_iteration):
                loss = self.abs_loss(images, delta, mask,
                                     Troj_Layer=Troj_Layer, Troj_next_Layer=Troj_next_Layer,
                                     Troj_Neuron=Troj_Neuron, Troj_next_Neuron=Troj_next_Neuron, Troj_size=Troj_size)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        tanh_delta = torch.tanh(delta).mul(0.5).add(0.5)
        con_mask = torch.tanh(mask) / 2.0 + 0.5
        con_mask = con_mask * self.nc_mask
        use_mask = con_mask.view(1, 1, self.h, self.w).repeat(1, 3, 1, 1)
        s_image = images.view(-1, 3, self.h, self.w)
        adv = s_image * (1 - use_mask) + tanh_delta * use_mask
        adv = torch.clamp(adv, 0.0, 1.0)

        acc, _ = self.model.accuracy(
            self.model.get_logits(adv), int(Troj_Label) * torch.ones(adv.shape[0], dtype=torch.long, device=self.model.device), topk=(1, 5))
        return (acc, adv.detach(), delta.detach(), con_mask.detach())

    def filter_img(self):
        h, w = self.dataset.n_dim
        mask = torch.zeros(h, w, dtype=torch.float)
        for i in range(h):
            for j in range(w):
                if j >= 2 and j < 8 and i >= 2 and i < 8:
                    mask[(i, j)] = 1
        return to_tensor(mask, non_blocking=False)

    def nc_filter_img(self) -> torch.Tensor:
        h, w = self.dataset.n_dim
        if self.use_mask:
            mask = torch.zeros(h, w, dtype=torch.float)
            for i in range(h):
                for j in range(w):
                    if not (j >= w * 0.25 and j < w * 0.75 and i >= h * 0.25 and i < h * 0.75):
                        mask[(i, j)] = 1
            mask = np.zeros((h, w), dtype=np.float32) + 1
        else:
            mask = np.zeros((h, w), dtype=np.float32) + 1
        return to_tensor(mask)

    def abs_loss(self, images, delta, mask, Troj_Layer, Troj_next_Layer, Troj_Neuron, Troj_next_Neuron, Troj_size, use_mask=None):
        if use_mask is None:
            tanh_delta = torch.tanh(delta).mul(0.5).add(0.5)
            con_mask = torch.tanh(mask) / 2.0 + 0.5
            con_mask = con_mask * self.nc_mask
            use_mask = con_mask.view(1, 1, self.h, self.w).repeat(1, 3, 1, 1)
        else:
            tanh_delta = delta
            con_mask = use_mask
            con_mask = con_mask * self.nc_mask
            use_mask = con_mask.view(1, 3, self.h, self.w)

        s_image = images.view(-1, 3, self.h, self.w)
        i_image = s_image * (1 - use_mask) + tanh_delta * use_mask

        tinners = self.model.get_layer(i_image, layer_output=Troj_Layer)
        ntinners = self.model.get_layer(i_image, layer_output=Troj_next_Layer)
        logits = self.model.get_layer(i_image, layer_output='logits')

        i_shape = tinners.shape
        ni_shape = ntinners.shape

        if len(i_shape) == 2:
            vloss1 = tinners[:, Troj_Neuron].sum()
            vloss2 = 0
            if Troj_Neuron > 0:
                vloss2 += tinners[:, :Troj_Neuron].sum()
            if Troj_Neuron < i_shape[(-1)] - 1:
                vloss2 += tinners[:, Troj_Neuron + 1:].sum()
        elif len(i_shape) == 4:
            vloss1 = tinners[:, :, :, Troj_Neuron].sum()
            vloss2 = 0
            if Troj_Neuron > 0:
                vloss2 += tinners[:, :, :, :Troj_Neuron].sum()
            if Troj_Neuron < i_shape[(-1)] - 1:
                vloss2 += tinners[:, :, :, Troj_Neuron + 1:].sum()
        # if len(ni_shape) == 2:
        #     relu_loss1 = ntinners[:, Troj_next_Neuron].sum()
        #     relu_loss2 = 0
        #     if Troj_Neuron > 0:
        #         relu_loss2 += ntinners[:, :Troj_next_Neuron].sum()
        #     if Troj_Neuron < i_shape[(-1)] - 1:
        #         relu_loss2 += ntinners[:, Troj_next_Neuron + 1:].sum()
        # if len(ni_shape) == 4:
        #     relu_loss1 = ntinners[:, :, :, Troj_next_Neuron].sum()
        #     relu_loss2 = 0
        #     if Troj_Neuron > 0:
        #         relu_loss2 += ntinners[:, :, :, :Troj_next_Neuron].sum()
        #     if Troj_Neuron < i_shape[(-1)] - 1:
        #         relu_loss2 += ntinners[:, :, :, Troj_next_Neuron + 1:].sum()
        tvloss = tv_loss(delta)
        loss = -vloss1 + 0.0001 * vloss2
        # loss = -vloss1 - relu_loss1 + 0.0001 * vloss2 + 0.0001 * relu_loss2
        mask_loss = con_mask.sum()
        mask_cond1 = mask_loss > float(Troj_size)
        mask_cond2 = mask_loss > 100.0
        if self.count_mask:
            mask_nz = len(F.relu(con_mask - 0.01).nonzero())
            mask_cond1 = mask_nz > Troj_size
            mask_cond2 = mask_nz > int((np.sqrt(Troj_size) + 2) ** 2)
        if mask_cond1:
            if mask_cond2:
                loss += 2 * self.re_mask_weight * mask_loss
            else:
                loss += self.re_mask_weight * mask_loss
        return loss
