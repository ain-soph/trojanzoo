# -*- coding: utf-8 -*-

from .badnet import BadNet

class TrojanNN(BadNet):
    
    name = 'trojannn'
    def __init__(self, preprocess_layer='features', neuron_num=2, batch_num=32, threshold=5, target_value=10, neuron_lr=0.015, neuron_epoch=20,
                 **kwargs):
        super().__init__(**kwargs)

        # Preprocess Parameters
        self.preprocess_layer = preprocess_layer
        self.neuron_num = neuron_num
        self.batch_num = batch_num
        self.threshold = threshold
        self.target_value = target_value
        self.neuron_lr = neuron_lr
        self.neuron_epoch = neuron_epoch

    def init_mark(self, output=None):
        output = self.get_output(output)
        shape = [self.model.dataset.n_channel,
                 self.model.dataset.n_dim[0], self.model.dataset.n_dim[1]]
        mark = read_img_as_tensor(self.mark_path)
        # org_mark_height = float(mark.shape[-2])
        # org_mark_width = float(mark.shape[-1])
        if self.mark_height == 0 and self.mark_width == 0:
            self.mark_height = int(self.mark_height_ratio*float(shape[-2]))
            self.mark_width = int(self.mark_width_ratio*float(shape[-1]))
        # assert self.mark_height != 0 and self.mark_width != 0
        if self.mark_height_offset is None:
            self.mark_height_offset = shape[-2]-self.mark_height
        if self.mark_width_offset is None:
            self.mark_width_offset = shape[-1]-self.mark_width
        # assert self.mark_height_offset is not None and self.mark_height_offset is not None

        mark = Image.fromarray(to_numpy(float2byte(mark)))
        mark = mark.resize(
            (self.mark_width, self.mark_height), Image.ANTIALIAS)
        mark = byte2float(to_numpy(mark)).unsqueeze(0)
        if 'preprocess' in output:
            print('Original Mark Shape: ', mark.shape)
        mark, mask, alpha_mask = self.mask_mark(
            mark, shape, self.mark_height_offset, self.mark_width_offset, edge_color=self.edge_color)
        mark = self.preprocess_mark(
            mark, mask, layer=self.preprocess_layer, output=self.output)  # random_init=False,
        if 'preprocess' in output:
            print('Present Mark Shape: ', mark.shape)
            print('Mask Shape: ', mask.shape)
        return mark, mask, alpha_mask

    # shape : channels, height, width
    # mark shape: 1, channels, height, width
    # mask shape: 1, channels, height, width
    # The mark shape may be smaller than the whole image. Fill the rest part as black, and return the mask and mark.
    def mask_mark(self, mark, shape, height_offset=None, width_offset=None, edge_color='auto', alpha=None):
        if alpha is None:
            alpha = self.alpha
        if len(mark.shape) == 2:
            mark = mark.unsqueeze(0)
        if len(mark.shape) == 3:
            mark = mark.unsqueeze(0)
        if shape[0] == 1 and mark.shape[1] == 3:
            mark = mark[:, 0].unsqueeze(1)
        if shape[0] != mark.shape[1]:
            print(shape)
            print(mark.shape)
            assert False

        t = torch.zeros(shape[0])
        if torch.is_tensor(edge_color):
            assert edge_color.shape.item() == shape[0]
            t = edge_color
        else:
            if edge_color == 'auto':
                _list = (to_tensor([mark[0, :, 0, :], mark[0, :, -1, :]]).view(shape[0], -1),
                         to_tensor([mark[0, :, :, 0], mark[0, :, :, -1]]).view(shape[0], -1))
                _list = torch.cat(_list).view(shape[0], -1)
                t = _list.mode(dim=-1)[0]
            if edge_color == 'black':
                t = to_tensor(torch.zeros(shape[0])).float()
            if edge_color == 'white':
                t = to_tensor(torch.ones(shape[0])).float()

        mask = to_tensor(torch.zeros(([1, 1, shape[1], shape[2]])))
        new_mark = to_tensor(torch.zeros(
            ([1, shape[0], shape[1], shape[2]])) - 1)
        for i in range(mark.shape[-2]):
            for j in range(mark.shape[-1]):
                if not mark[0, :, i, j].view(-1).equal(t):
                    mask[0, 0, height_offset + i, width_offset + j] = 1
                    new_mark[0, :, height_offset + i,
                             width_offset + j] = mark[0, :, i, j]
        mask = mask.repeat(1, shape[0], 1, 1)
        self.mask_pixel_num = mask.sum()
        print('mask_pixel_num: ', self.mask_pixel_num)
        alpha_mask = (mask*(1-alpha))
        return new_mark.detach(), mask.detach(), alpha_mask.detach()

    # Give the mark init values for non transparent pixels.
    def random_init_mark(self, mark, mask):
        init_mark = to_tensor(to_valid_img(torch.randn_like(mark)))
        zeros = to_tensor(torch.zeros_like(mark))
        init_mark = torch.where(mask == 1, init_mark, zeros-1)
        return init_mark.detach()

    # add mark to the Image with mask.
    def add_mark(self, X, mark, _mask, detach=True, original=False):
        if original:
            return X
        result = to_tensor(X*(1-_mask)+mark*_mask)
        if detach:
            result = result.detach()
        return result

    # get the neuron idx for preprocess.
    def get_neuron_idx(self):
        fm_shape = to_tensor(torch.Tensor([0]))
        for i, data in enumerate(self.model.dataset.loader['train']):
            _input, _label = self.model.get_data(data)
            fm_shape = to_tensor(self.model.get_layer(
                _input, layer_output=self.preprocess_layer).shape)
            # layer may vary. Using get_fm
            break
        _result = to_tensor(torch.zeros(fm_shape[1]))
        for i, data in enumerate(self.model.dataset.loader['train']):
            if self.batch_num is not None and self.batch_num != 0 and i >= self.batch_num:
                break
            _input, _label = self.model.get_data(data)
            fm = to_tensor(self.model.get_layer(
                _input, layer_output=self.preprocess_layer))
            # layer may vary. Using get_fm
            # ones = to_tensor(torch.ones_like(fm))
            # result = torch.where(fm == 0, ones, ones-1)
            result = fm.view(fm.shape[0], fm.shape[1], -1)
            result = result.mean(dim=2).mean(dim=0)
            _result += result.detach()
        return _result.argsort(descending=False)

    def get_neuron_value(self, x, neuron_idx):

        fm = to_tensor(self.model.get_layer(x, layer_output=self.preprocess_layer)
                       )[:, neuron_idx].mean()
        return fm

    # train the mark to activate the least-used neurons.
    def preprocess_mark(self, mark, mask, random_init=True, neuron_idx=None, output=None, **kwargs):
        if output is None:
            output = self.output

        if random_init:
            mark = self.random_init_mark(mark, mask)
        if neuron_idx is None:
            neuron_idx = self.get_neuron_idx()
        neuron_idx = neuron_idx[:self.neuron_num]
        print("Neuron Value Before Preprocessing: ",
              self.get_neuron_value(mark, neuron_idx))
        # print("max: ",_model.get_layer(mark,layer='layer3').max())
        _temp = self.model.get_layer(mark, layer_output=self.preprocess_layer)
        print('feature shape: ', _temp.shape)
        # zero_num = torch.where(_temp == 0, torch.ones_like(
        #     _temp), torch.zeros_like(_temp))
        # print(zero_num.squeeze()[neuron_idx].sum())

        def loss_func(X):
            loss = self.model.get_layer(X, layer_output=self.preprocess_layer)[
                :, neuron_idx]-self.target_value
            loss = loss.pow(2).view(loss.shape[0], loss.shape[1], -1).mean()
            return loss

        mark = to_tensor(mark.detach())
        x = deepcopy(mark)
        noise = to_tensor(torch.zeros_like(x))
        for _iter in range(self.neuron_epoch):
            cost = loss_func(x)
            if cost < self.threshold:
                break

            x, _ = self.module.pgd.perturb(
                mark, noise, alpha=self.neuron_lr, epsilon=1.0, iteration=1, loss_func=loss_func, output=0)
        _temp = self.model.get_layer(mark, layer_output=self.preprocess_layer)
        # zero_num = torch.where(_temp == 0, torch.ones_like(
        #     _temp), torch.zeros_like(_temp))
        # print(zero_num.squeeze()[neuron_idx].sum())
        print("Neuron Value After Preprocessing: ",
              self.get_neuron_value(x, neuron_idx))

        return x.detach()

    def watermark_pgd(self, mark, mask, alpha_mask, target=None, output=None, **kwargs):

        if output is None:
            output = self.output
        # noise = mark.detach()-org_mark.detach()
        noise = 0
        prev = 1
        init = True

        pgd_alpha = self.module.pgd.alpha

        _mask = alpha_mask*mask

        for _iter in range(self.module.pgd.iteration):
            for i, data in enumerate(self.model.dataset.loader['train']):
                _input, _label = self.model.get_data(data)
                _X = self.add_mark(_input.detach(), mark.detach(), _mask)

                _target = to_tensor(target).repeat(_label.shape[0]).detach()
                _X.requires_grad = True

                def loss_func(Xj):
                    loss = self.model.loss(Xj, _target)
                    return loss
                if self.module.pgd.mode == 'white':
                    loss = loss_func(_X)
                    grad = torch.autograd.grad(loss, _X)[0]
                elif self.module.pgd.mode == 'black':
                    grad = self.cal_gradient(loss_func, _X)
                else:
                    print('Value of Parameter "mode" should be "white" or "black"!')
                    sys.exit(-1)
                grad = grad.detach().mean(dim=0, keepdim=True)
                noise = (noise/4 - grad)*mask
                if init:
                    prev = grad.view(-1).norm(p=2)
                present = grad.view(-1).norm(p=2)
                pgd_alpha = pgd_alpha*present/prev

                mark = to_valid_img(mark.detach()+pgd_alpha *
                                    noise).detach()
                # noise = to_valid_img(
                #     noise, min=-self.pgd_epsilon, max=self.pgd_epsilon)

                # mark = to_valid_img(org_mark.detach() + noise.detach())
        return mark, mask, alpha_mask

    def watermark_adam(self, mark, mask, alpha_mask, target=None, output=None, **kwargs):
        if output is None:
            output = self.output
        # noise = mark.detach()-org_mark.detach()
        noise = 0

        atanh_mark = arctanh(mark*2-1).detach()
        atanh_mark.requires_grad = True
        atanh_alpha_mask = arctanh(alpha_mask*2-1).detach()
        atanh_alpha_mask.requires_grad = True
        optimizer = optim.Adam(
            [atanh_mark, atanh_alpha_mask], lr=0.01, betas=(0.5, 0.9))
        optimizer.zero_grad()

        for _iter in range(self.module.pgd.iteration):
            loss_cls = 0.0
            loss_reg = 0.0
            count = 0

            for i, data in enumerate(tqdm(self.model.dataset.loader['train'])):
                _input, _label = self.model.get_data(data)
                X = to_tensor(_input)
                X.requires_grad = True

                _mark = torch.tanh(atanh_mark).add(1).div(2)
                _mask = mask*torch.tanh(atanh_alpha_mask).add(1).div(2)
                _X = self.add_mark(X, _mark, _mask, detach=False)
                _target = to_tensor(target).repeat(_label.shape[0]).detach()

                loss = self.model.loss(_X, _target)
                # loss = to_tensor(torch.tensor(0.0))
                h = to_tensor(torch.Tensor([0.0]))
                loss_cls += float(loss)
                if self.adapt == 'neural_cleanse' or self.adapt == 'both':
                    h += _mask.norm(p=1)/self.mask_pixel_num
                    loss += h

                # if adapt=='strip' or adapt=='both':
                #     h = -strip.detect(_X)[1]
                #     loss += h * 0.1
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss_reg += float(h)
                count += 1
            loss_cls /= count
            loss_reg /= count

            print('classification loss: ', loss_cls)
            print('regularization loss: ', loss_reg)
        mark = torch.tanh(atanh_mark).add(1).div(2)
        alpha_mask = torch.tanh(atanh_alpha_mask).add(1).div(2)

        return mark, mask, alpha_mask

    def retrain(self, retrain_epoch: int = None, target: torch.LongTensor = None, percent=None,
                watermark: torch.FloatTensor = None, mask: torch.FloatTensor = None,
                train_opt='full', lr_scheduler=False, validate_interval=10, parallel=True, **kwargs):
        if retrain_epoch is None:
            retrain_epoch = self.retrain_epoch
        if target is None:
            target = self.target_class
        if percent is None:
            percent = self.percent

        optimizer = self.model.define_optimizer(
            train_opt=train_opt, lr_scheduler=lr_scheduler, **kwargs)
        optimizer.zero_grad()
        _lr_scheduler = None
        if lr_scheduler:
            _lr_scheduler = optimizer
            optimizer = _lr_scheduler.optimizer
        self.model.train()

        _lambda = 0.2 if self.model.dataset.name == 'gtsrb' else 0.6

        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        if self.adapt == 'abs':
            _dict = np.load(self.abs.seed_path, allow_pickle=True).item()
            fxs, fys = _dict['x'], _dict['y']
            fxs = to_tensor(fxs, dtype='float')/255
            fys = to_tensor(fys, dtype='long')
            xs = fxs[:20]
            ys = fys[:20]
            test_xs = fxs[20:]
            test_ys = fys[20:]
            all_ps = self.abs.sample_neuron(xs, ys)
            neuron_dict = self.abs.find_min_max(all_ps)
        for _epoch in range(retrain_epoch):
            entropy = 0.0
            counter = 0
            losses.reset()
            top1.reset()
            top5.reset()
            for i, data in enumerate(tqdm(self.model.dataset.loader['train'])):
                X, Y = self.model.get_data(data, mode='train')

                loss = to_tensor(torch.Tensor([0.0]))
                if watermark is not None and random.uniform(0, 1) < percent:
                    batch_size = X.shape[0]
                    batch_target = repeat_to_batch(
                        to_tensor(target).squeeze(), batch_size)
                    X2 = deepcopy(X).detach()
                    X2 = self.add_mark(X2, watermark, mask, self.alpha)
                    if self.adapt == 'strip' or self.adapt == 'both':
                        h = -self.strip.detect(X2)[1]
                        loss += h*_lambda
                        entropy += float(h)
                        counter += 1
                    if self.adapt == 'abs':
                        abs_loss = self.abs.re_mask_loss(neuron_dict,
                                                         xs, watermark, mask)
                        loss -= 1e-4*abs_loss

                    X = torch.cat((X, X2))
                    Y = torch.cat((Y, batch_target))
                _output = self.model.get_logits(X, parallel=parallel)
                loss += self.model.criterion(_output, Y)
                loss.backward(retain_graph=True)
                optimizer.step()
                optimizer.zero_grad()

                acc1, acc5 = self.model.accuracy(_output, Y, topk=(1, 5))
                losses.update(loss.item(), Y.size(0))
                top1.update(acc1[0], Y.size(0))
                top5.update(acc5[0], Y.size(0))

            print(('Epoch: [%d/%d],' % (_epoch+1, retrain_epoch)).ljust(25, ' ') +
                  'Loss: %.4f,\tTop1 Acc: %.3f,\tTop5 Acc: %.3f' % (losses.avg, top1.avg, top5.avg))
            if lr_scheduler:
                _lr_scheduler.step()

            if validate_interval != 0:
                if (_epoch+1) % validate_interval == 0 or _epoch == retrain_epoch - 1:
                    _, cur_acc, _ = self.model._validate()
                    self.model.train()
                    print('---------------------------------------------------')
            if self.adapt == 'strip' or self.adapt == 'both':
                print('entropy: ', entropy/counter)
        self.model.zero_grad()
        self.model.eval()

    def target_class_proportion(self, mark: torch.FloatTensor = None, mask: torch.FloatTensor = None, alpha_mask: torch.FloatTensor = None, target: int = None, original=False):
        if mark is None:
            mark = self.mark
        if mask is None:
            mask = self.mask
        if alpha_mask is None:
            alpha_mask = self.alpha_mask
        if target is None:
            target = self.target_class
        correct = 0
        total = 0

        _mask = mask*alpha_mask
        for i, data in enumerate(self.model.dataset.loader['valid']):
            _input, _label = self.model.get_data(data, mode='train')
            X = self.add_mark(_input, mark, _mask, original=original)
            ones = to_tensor(torch.ones([X.shape[0]]))
            result = self.model(X).argmax(dim=-1)

            num_target = torch.where(_label == target, ones, ones-1)
            cor = torch.where(result == target, ones, ones-1)
            rep = cor+num_target

            repeat = torch.where(rep == to_tensor(2.), ones, ones-1)

            total += X.shape[0]-num_target.sum()
            correct += cor.sum()-repeat.sum()

        return float(correct)/total

    def target_class_confidence(self, mark: torch.FloatTensor = None, mask: torch.FloatTensor = None, alpha_mask: torch.FloatTensor = None, target: int = None, original=False):
        if mark is None:
            mark = self.mark
        if mask is None:
            mask = self.mask
        if alpha_mask is None:
            alpha_mask = self.alpha_mask
        if target is None:
            target = self.target_class

        _mask = mask*alpha_mask

        result_list = []
        for i, data in enumerate(self.model.dataset.loader['valid']):
            _input, _label = self.model.get_data(data, mode='train')
            X = self.add_mark(_input, mark, _mask, original=original)
            # idx = []
            # for i in range(len(Y)):
            #     if Y[i] != target:
            #         idx.append(i)
            # X = X[i]
            # Y = Y[i]
            # X = self.add_mark(X, mark, _mask, original=original)
            result = self.model.get_prob(X)[:, target]
            result_list.extend(result.detach().cpu().tolist())
        return result_list
