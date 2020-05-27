# -*- coding: utf-8 -*-

from package.utils.utils import to_numpy, to_tensor, save_tensor_as_img, save_numpy_as_img
from package.model.image_cnn import Image_CNN
from package.imports.universal import *

from copy import deepcopy
import pickle


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :]-x[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:]-x[:, :, :, :w_x-1]), 2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self, t):
        return t.size()[1]*t.size()[2]*t.size()[3]


tv_loss = TVLoss()


class ABS():

    def __init__(self, model: Image_CNN = None, re_iteration=1000, **kwargs):
        self.model = model
        self.h = 32
        self.w = 32

        self.use_mask = True
        self.count_mask = True
        self.tdname = 'temp'

        self.nc_mask = self.nc_filter_img(
            self.h, self.w, use_mask=self.use_mask)

        # Neural Sampling
        self.samp_k = 1
        self.same_range = False
        self.n_samples = 5
        self.samp_batch_size = 8
        self.top_n_neurons = 20

        # ReMask
        self.max_troj_size = 16
        self.mask_multi_start = 1
        self.filter_multi_start = 1
        self.re_mask_lr = 0.1
        self.re_mask_weight = 500
        self.re_iteration = re_iteration

        self.seed_path = './data/seed.npy'
        self.result_dir = './result/'
        path = self.result_dir+'imgs/'
        if not os.path.exists(path):
            os.makedirs(path)
        path = self.result_dir+'masks/'
        if not os.path.exists(path):
            os.makedirs(path)
        path = self.result_dir+'deltas/'
        if not os.path.exists(path):
            os.makedirs(path)

    def trojan_prob(self, weights_file: str):
        _dict = np.load(self.seed_path, allow_pickle=True).item()
        fxs, fys = _dict['x'], _dict['y']
        fxs = to_tensor(fxs, dtype='float')/255
        fys = to_tensor(fys, dtype='long')
        xs = fxs[:20]
        ys = fys[:20]
        test_xs = fxs[20:]
        test_ys = fys[20:]
        self.model.load(weights_file)
        all_ps = self.sample_neuron(xs, ys)
        neuron_dict = self.find_min_max(all_ps)
        results = self.re_mask(neuron_dict, xs, weights_file)
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

#-----------------------Test Mask---------------------------------#

    def stamp(self, n_img, delta, mask):
        mask0 = self.nc_filter_img(self.h, self.w, use_mask=self.use_mask)
        mask = mask * mask0
        r_img = n_img * (1 - mask) + delta * mask
        return r_img

    def test_mask(self, weights_file, test_xs, result):
        rimg, rdelta, rmask, tlabel = result[:4]
        self.model.load(weights_file)
        t_images = self.stamp(test_xs, rdelta, rmask)
        for i in range(len(t_images)):
            save_numpy_as_img(
                self.tdname + '/{0}.png'.format(i), t_images[i])

        yt = int(tlabel)*torch.ones(len(t_images),
                                    dtype=torch.long, device=self.model.device)
        acc, _ = self.model.accuracy(self.model(t_images), yt, topk=(1, 5))
        return acc

#----------------------------------------------------------------#


#-----------------------Neural Sample----------------------------#

    # find the maximum value of each layer in the image batch


    def check_values(self, _input):
        maxes = {}
        layer_output = self.model.get_all_layer(_input)
        for layer in layer_output.keys():
            if 'pool' not in layer:
                maxes[layer] = layer_output[layer].max()
        return maxes

    def sample_neuron(self, _input, _label):
        all_ps = {}
        batch_size = _input.shape[0]

        maxes = self.check_values(_input)

        layer_output = self.model.get_all_layer(_input)
        layer_name_list = self.model.get_layer_name()
        for layer in layer_name_list:
            if 'pool' in layer or layer in ['features', 'classifier', 'logits', 'output']:
                continue
            neuron_num = layer_output[layer].shape[-1]
            if self.same_range:
                vs = [i * self.samp_k for i in range(self.n_samples)]
            else:
                tr = self.samp_k * float(maxes[layer]) / self.n_samples
                vs = tr*np.arange(self.n_samples)
            neuron_batch_number = neuron_num // self.samp_batch_size

            for neuron_batch_idx in range(neuron_batch_number):
                l_h_t = []
                for neuron in range(self.samp_batch_size):
                    if len(layer_output[layer].shape) == 4:
                        h_t = layer_output[layer].repeat(
                            self.n_samples, 1, 1, 1)
                    elif len(layer_output[layer].shape) == 2:
                        h_t = layer_output[layer].repeat(self.n_samples, 1)
                    else:
                        print('layer output shape: ',
                              layer_output[layer].shape)
                        raise ValueError()
                    for i, v in enumerate(vs):
                        if len(layer_output[layer].shape) == 4:
                            h_t[i * batch_size:(i + 1) * batch_size, :, :,
                                neuron + neuron_batch_idx * self.samp_batch_size] = v
                        elif len(layer_output[layer].shape) == 2:
                            h_t[i * batch_size:(i + 1) * batch_size,
                                neuron + neuron_batch_idx * self.samp_batch_size] = v
                        else:
                            print('layer output shape: ',
                                  layer_output[layer].shape)
                            raise ValueError()
                    l_h_t.append(h_t)

                # (samp_batch_size * n_samples * batch_size, ...)
                f_h_t = torch.cat(l_h_t)
                fps = self.model.get_layer(f_h_t, layer_input=layer)
                for neuron in range(self.samp_batch_size):
                    # (n_samples * batch_size, num_classes)
                    tps = fps[neuron * self.n_samples *
                              batch_size: (neuron + 1) * self.n_samples * batch_size]
                    for idx_img in range(batch_size):
                        img_name = ('{0}_0.jpg').format(_label[idx_img])
                        ps_key = ('{0}_{1}_{2}').format(
                            img_name, layer, neuron + neuron_batch_idx * self.samp_batch_size)
                        # (n_samples, num_classes)
                        ps = np.array([to_numpy(tps[(idx_img + batch_size * i)])
                                       for i in range(self.n_samples)])
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
                vs.append(np.amax(all_ps[k][l][all_ps[k].shape[1] // 5:]) -
                          np.amin(all_ps[k][l][:all_ps[k].shape[1] // 5]))

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

#----------------------------------------------------------------#


#-------------------------ReMask---------------------------------#

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
            RE_img = (self.result_dir+'imgs/{0}_model_{1}_{2}_{3}_{4}.png').format(weights_file.split(
                '/')[(-1)][:-3], Troj_Layer, Troj_Neuron, Troj_size, Troj_Label)
            RE_mask = (self.result_dir+'masks/{0}_model_{1}_{2}_{3}_{4}.png').format(weights_file.split(
                '/')[(-1)][:-3], Troj_Layer, Troj_Neuron, Troj_size, Troj_Label)
            RE_delta = (self.result_dir+'deltas/{0}_model_{1}_{2}_{3}_{4}.npy').format(weights_file.split(
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

        self.model.load(weights_file)
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
            self.model.get_logits(adv), int(Troj_Label)*torch.ones(adv.shape[0], dtype=torch.long, device=self.model.device), topk=(1, 5))
        return (acc, adv.detach(), delta.detach(), con_mask.detach())

    @staticmethod
    def filter_img(h, w):
        mask = np.zeros((h, w), dtype=np.float32)
        for i in range(h):
            for j in range(w):
                if j >= 2 and j < 8 and i >= 2 and i < 8:
                    mask[(i, j)] = 1
        return to_tensor(mask)

    @staticmethod
    def nc_filter_img(h, w, use_mask=True):
        if use_mask:
            mask = np.zeros((h, w), dtype=np.float32)
            for i in range(h):
                for j in range(w):
                    if not (j >= w * 1 / 4.0 and j < w * 3 / 4.0 and i >= h * 1 / 4.0 and i < h * 3 / 4.0):
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
        mask_nz = len(F.relu(con_mask - 0.01).nonzero())
        if self.count_mask:
            mask_cond1 = mask_nz > Troj_size
            mask_cond2 = mask_nz > int((np.sqrt(Troj_size) + 2) ** 2)
        if mask_cond1:
            if mask_cond2:
                loss += 1000*mask_loss
            else:
                loss += 500 * mask_loss
        return loss
#----------------------------------------------------------------#
