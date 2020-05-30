# -*- coding: utf-8 -*-
from trojanzoo.attack.attack import Attack
from trojanzoo.imports import *
from trojanzoo.utils import *
import random

from copy import deepcopy


class HiddenBackdoor(Attack):

    name = 'hiddenbackdoor'

    def __init__(self,
                 target_class=0,
                 mask_path='',
                 mark_path='',
                 model_path='',
                 lr=0.01,
                 decay=False,
                 decay_ratio=0.95,
                 decay_iteration=2000,
                 epsilon=16,
                 retrain_epoch=10,
                 poison_generation_iteration=10000,
                 batch_num=100,
                 poisoned_image_num=100,
                 original=False,
                 train_opt='partial',
                 lr_scheduler=False,
                 preprocess_layer='features',
                 validate_interval=10,
                 parallel=True,
                 folder_path='./hidden_backdoor_result',
                 use_gpu=True,
                 **kwargs):
        """
        
        HiddenBackdoor attack is different with trojan nn(References: https://docs.lib.purdue.edu/cgi/viewcontent.cgi?article=2782&context=cstech),the mark and mask is designated and stable, we continue these used in paper(References: https://arxiv.org/abs/1910.00033).
        self.mask_path,self.mark_path,self.model_path must be set  corresponding to the identical dataset and set initially.
        """
        super(HiddenBackdoor, self).__init__(**kwargs)

        # experiment parameter

        self.lr = lr
        self.decay = decay
        self.decay_ratio = decay_ratio
        self.decay_iteration = decay_iteration
        self.epsilon = epsilon
        self.retrain_epoch = retrain_epoch
        self.batch_num = batch_num
        self.poison_generation_iteration = poison_generation_epoch
        self.poisoned_image_num = poisoned_image_num
        self.lr_scheduler = lr_scheduler
        self.validate_interval = validate_interval
        self.parallel = parallel
        self.train_opt = train_opt

        # poison sample parameter
        self.mask_path = mask_path
        self.mark_path = mark_path
        self.target_class = target_class

        self.original = original
        self.model_path = model_path
        self.folder_path = folder_path
        self.preprocess_layer = preprocess_layer
        self.use_gpu = use_gpu

    def load_mark_mask(self, mark_path: str = None, mask_path: str = None):
        """
        Load mark and mask existed in the form of npy, according to the paper, they are set initially. The mark, mask, model all should be corresponding to the identical dataset.
        :param mark_path: the file path of the mark
        :param mask_path: the file path of the mask
        :returns self.mark & self.mask
        """
        if mark_path is None:
            mark_path = self.mark_path
        if mask_path is None:
            mask_path = self.mask_path
        self.mark = to_tensor(np.load(mark_path))
        self.mask = to_tensor(np.load(mask_path))

    def load_model(self, model_path: str = None):
        """
        Load the pretrained model. The mark, mask, model all should be corresponding to the identical dataset.
        :param model_path: the file path of the model
        :returns self.model
        """
        if model_path is None:
            model_path = self.model_path
        self.model.load_pretrained_weights(model_path)

    def add_mark(self, X, detach=True, original=False):
        """
        Add the mark to initial sample.
        :param X: the file path of the model
        :param detach: whether to detach the output from the computationnal graph
        :param original:whether to add the trigger to X
        :returns: result: the processed sample (with or without mark)
        """
        if original:
            return X
        result = to_tensor(X * (1 - self.mask) + self.mark * self.mask)
        if detach:
            result = result.detach()
        return result

    def target_class_proportion(self, target: int = None, original=False):
        """
        Compute the proportion of the samples from target class, with or without mark, in valid set.
        :param target: the target class
        :param original: whether to add the trigger to samples
        :returns: float(correct)/total: the proportion of the samples from target class
        """

        if target is None:
            target = self.target_class

        correct = 0
        total = 0

        for i, data in enumerate(self.model.dataset.loader['valid']):
            _input, _label = self.model.get_data(data, mode='train')
            X = self.add_mark(_input, original=original)
            ones = to_tensor(torch.ones([X.shape[0]]))
            result = self.model(X).argmax(dim=-1)

            num_target = torch.where(_label == target, ones, ones - 1)
            cor = torch.where(result == target, ones, ones - 1)
            rep = cor + num_target

            repeat = torch.where(rep == to_tensor(2.), ones, ones - 1)

            total += X.shape[0] - num_target.sum()
            correct += cor.sum() - repeat.sum()

        return float(correct) / total

    def target_class_confidence(self, target: int = None, original=False):
        """
        Compute the confidence of the samples, with or without mark, in valid set classfied as target class.
        :param target: the target class
        :param original:whether to add the trigger to samples
        :returns: result_list: the confidence result list of samples in valid set classfied as target class.
        """
        if target is None:
            target = self.target_class

        result_list = []

        for i, data in enumerate(self.model.dataset.loader['valid']):
            _input, _label = self.model.get_data(data, mode='train')
            X = self.add_mark(_input, original=original)
            result = self.model.get_prob(X)[:, target]
            result_list.extend(result.detach().cpu().tolist())
        return result_list

    def perturb(self,
                output=None,
                mark_path: str = None,
                mask_path: str = None,
                model_path: str = None,
                target: int = None,
                train_opt=None,
                lr_scheduler=None,
                retrain_epoch: int = None,
                validate_interval: int = None,
                parallel=None,
                poisoned_image_num: int = None,
                epsilon: int = None,
                preprocess_layer=None,
                poison_generation_iteration: int = None,
                decay=None,
                decay_ratio=None,
                decay_iteration=None):
        """
        Test the performance of poisoned model on normal samples and samples patched by mark. Save the poisoned model and confidence list.
        :param output: output added by hand, such as some notes
        :param mark_path: the file path of the mark
        :param mask_path: the file path of the mask
        :param model_path: the file path of the model
        :param target: the target class
        :param train_opt: specify whether finetune the whole model or only part
        :param lr_scheduler: specify whether to adjust learning rate 
        :param retrain_epoch: how many epoches that the model needs to retrained 
        :param validate_interval: the interval epoch needed to validate the model
        :param parallel: specify whether to parallel process data
        :param poisoned_image_num: the number of poisoned images
        :param epsilon: the threshold in pixel space to ensure the poisoned image is not visually distinguishable from the target image
        :param preprocess_layer: the chosen specific layer that on which the feature space of source images patched by trigger is close to poisoned images 
        :param poison_generation_iteration: the iteration times used to generate one poison image
        :param decay: specify whether the learning rate decays with iteraion times
        :param decay_ratio: specify the learning rate decay proportion
        :param decay_iteration: specify the number of iteration time interval, the learning rate will decays once
        :returns some results and the poisoned model
        """
        if mark_path is None:
            mark_path = self.mark_path
        if mask_path is None:
            mask_path = self.mask_path
        if model_path is None:
            model_path = self.model_path
        if target is None:
            target = self.target_class
        if train_opt is None:
            train_opt = self.train_opt
        if lr_scheduler is None:
            lr_scheduler = self.lr_scheduler
        if retrain_epoch is None:
            retrain_epoch = self.retrain_epoch
        if validate_interval is None:
            validate_interval = self.validate_interval

        if parallel is None:
            parallel = self.parallel
        if poisoned_image_num is None:
            poisoned_image_num = self.poisoned_image_num
        if epsilon is None:
            epsilon = self.epsilon
        if preprocess_layer is None:
            preprocess_layer = self.preprocess_layer
        if poison_generation_iteration is None:
            poison_generation_iteration = self.poison_generation_iteration
        if decay is None:
            decay = self.decay
        if decay_ratio is None:
            decay_ratio = self.decay_ratio
        if decay_iteration is None:
            decay_iteration = self.decay_iteration

        output = self.get_output(output)

        self.load_mark_mask(mark_path, mask_path)
        self.load_model(model_path)

        self.retrain(train_opt=train_opt,
                     lr_scheduler=lr_scheduler,
                     retrain_epoch=retrain_epoch,
                     target=target,
                     validate_interval=validate_interval,
                     parallel=parallel,
                     poisoned_image_num=poison_generation_iteration,
                     epsilon=epsilon,
                     preprocess_layer=preprocess_layer,
                     poison_generation_iteration=poison_generation_iteration,
                     decay=decay,
                     decay_ratio=decay_ratio,
                     decay_iteration=decay_iteration)

        succ_rate = self.target_class_proportion(target)
        mis_rate = self.target_class_proportion(target, original=True)
        confidence_list = self.target_class_confidence(target)
        print('Succ Rate: ', succ_rate)
        print('Mis Rate: ', mis_rate)
        print('Confidence: ', np.mean(confidence_list))
        np.save(self.folder_path + '/confidence.npy', confidence_list)
        self.model.save_weights(self.folder_path +
                                '/hidden_backdoor_poisoned.pth',
                                full=True)
        print('model and confidence are saved at %s!' % (self.folder_path))

    def adjust_lr(self,
                  iteration,
                  decay=None,
                  decay_ratio=None,
                  decay_iteration=None):
        """
        Adjust learning_rate according to iteration.
        :param iteration: the number of iterations, especially in the process of generating poisoned image
        :param decay: specify whether the learning rate decays with iteraion times
        :param decay_ratio: specify the learning rate decay proportion
        :param decay_iteration: specify the number of iteration time interval, the learning rate will decays once
        :returns: lr: the adjusted learning_rate
        """
        if decay is None:
            decay = self.decay
        if decay_ratio is None:
            decay_ratio = self.decay_ratio
        if decay_iteration is None:
            decay_iteration = self.decay_iteration

        if decay == True:
            lr = self.lr
            lr = lr * (decay_ratio**(iteration // decay_iteration))
            return lr
        else:
            return self.lr

    def generate_poisoned_image(self,
                                source_image,
                                target_image,
                                epsilon: int = None,
                                preprocess_layer=None,
                                poison_generation_iteration: int = None,
                                decay=None,
                                decay_ratio=None,
                                decay_iteration=None):
        """
        According to the sampled target images and the sampled source images patched by the trigger ,modify the target image to generate poison images ,that is close to images of target category in pixel space and also close to source images patched by the trigger in feature space.
        :param source_image: self.poisoned_image_num source images, other than target category, sampled from dataloader['train'] 
        :param target_image: self.poisoned_image_num target images sampled from the images of target category in dataloader['train']
        :param epsilon: the threshold in pixel space to ensure the poisoned image is not visually distinguishable from the target image
        :param preprocess_layer: the chosen specific layer that on which the feature space of source images patched by trigger is close to poisoned images 
        :param poison_generation_iteration: the iteration times used to generate one poison image
        :param decay: specify whether the learning rate decays with iteraion times
        :param decay_ratio: specify the learning rate decay proportion
        :param decay_iteration: specify the number of iteration time interval, the learning rate will decays once 
        :returns: generated_poisoned_image: the self.poisoned_image_num generated poisoned image
        """
        if epsilon is None:
            epsilon = self.epsilon
        if preprocess_layer is None:
            preprocess_layer = self.preprocess_layer
        if poison_generation_iteration is None:
            poison_generation_iteration = self.poison_generation_iteration
        if decay is None:
            decay = self.decay
        if decay_ratio is None:
            decay_ratio = self.decay_ratio
        if decay_iteration is None:
            decay_iteration = self.decay_iteration
        if torch.cuda.is_available() and self.use_gpu == True:
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        source_image = source_image.to(self.device)
        target_image = target_image.to(self.device)
        generated_poisoned_image = torch.zeros_like(source_image)
        pert = nn.Parameter(
            torch.zeros_like(target_image, requires_grad=True).to(self.device))

        source_image = self.add_mark(source_image)
        output1 = self.model(source_image)

        feat1 = to_tensor(
            self.model.get_layer(
                source_image, layer_output=preprocess_layer)).detach().clone()

        for j in range(poison_generation_iteration):
            output2 = model(target_image + pert)
            feat2 = to_tensor(
                self.model.get_layer(
                    target_image,
                    layer_output=preprocess_layer)).detach().clone()
            feat11 = feat1.clone()
            dist = torch.cdist(feat1, feat2)
            for _ in range(feat2.size(0)):
                dist_min_index = (dist == torch.min(dist)).nonzero().squeeze()
                feat1[dist_min_index[1]] = feat11[dist_min_index[0]]
                dist[dist_min_index[0], dist_min_index[1]] = 1e5

            loss1 = ((feat1 - feat2)**2).sum(
                dim=1
            )  #  Decrease the distance between sourced images patched by trigger and target images
            loss = loss1.sum()
            losses.update(loss.item(), source_image.size(0))
            loss.backward()
            lr = self.adjust_lr(iteration=j,
                                decay=decay,
                                decay_ratio=decay_ratio,
                                decay_iteration=decay_iteration)
            pert = pert - lr * pert.grad
            pert = torch.clamp(pert, -(epsilon / 255.0),
                               epsilon / 255.0).detach_()
            pert = pert + target_image
            pert = pert.clamp(0, 1)  # restrict the pixel value range resonable
            if j % 100 == 0:
                print(
                    "Epoch: {:2d} | i: {} | iter: {:5d} | LR: {:2.4f} | Loss Val: {:5.3f} | Loss Avg: {:5.3f}"
                    .format(epoch, i, j, lr, losses.val, losses.avg))
            if loss1.max().item() < 10 or j == (poison_generation_iteration -
                                                1):
                for k in range(target_image.size(0)):
                    input2_pert = (pert[k].clone())
                    generated_poisoned_image[k] = input2_pert
                break

            pert = pert - target_image
            pert.requires_grad = True
        return generated_poisoned_image

    def sample_size(self):
        """
        Get the shape of single image ( 2d or 3d) 
        :returns: list(tuple(X.shape)): the shape of single image existed in the form of list
        """
        for i, data in enumerate(tqdm(self.model.dataset.loader['train'])):
            X, Y = self.model.get_data(data, mode='train')
            break
        return list(tuple(X.shape))

    def sample_target(self,
                      target: int = None,
                      poisoned_image_num: int = None):
        """
        Get self.poisoned_image_num samples from target category in  self.model.dataset.loader['train']
        :param target: the target class
        :param poisoned_image_num: the number of poisoned images
        :returns: target_sample: sampled images from target class
        """
        if target is None:
            target = self.target_class
        if poisoned_image_num is None:
            poisoned_image_num = self.poisoned_image_num

        num_target = 0
        sample_shape = self.sample_size()
        if len(sample_shape) > 3:
            target_sample = torch.zeros([
                poisoned_image_num, sample_shape[1], sample_shape[2],
                sample_shape[3]
            ])
        else:
            target_sample = torch.zeros(
                [poisoned_image_num, sample_shape[1], sample_shape[2]])

        for i, data in enumerate(tqdm(self.model.dataset.loader['train'])):
            X, Y = self.model.get_data(data, mode='train')
            for j in range(len(Y)):
                if Y[j, 0].item() == target:
                    target_sample[num_target] = X[j]
                    num_target += 1
                if num_target > poisoned_image_num:
                    break
            if num_target > poisoned_image_num:
                break
        return target_sample

    def sample_non_target(self,
                          target: int = None,
                          poisoned_image_num: int = None):
        """
        Get self.poisoned_image_num samples other than target category in  self.model.dataset.loader['train']
        :param target: the target class
        :param poisoned_image_num: the number of poisoned images
        :returns: non_target_sample: self.poisoned_image_num source images
        """
        if target is None:
            target = self.target_class
        if poisoned_image_num is None:
            poisoned_image_num = self.poisoned_image_num

        num_non_target = 0
        sample_shape = self.sample_size()
        if len(sample_shape) > 3:
            non_target_sample = torch.zeros([
                poisoned_image_num, sample_shape[1], sample_shape[2],
                sample_shape[3]
            ])
        else:
            non_target_sample = torch.zeros(
                [poisoned_image_num, sample_shape[1], sample_shape[2]])

        for i, data in enumerate(tqdm(self.model.dataset.loader['train'])):
            X, Y = self.model.get_data(data, mode='train')
            for j in range(len(Y)):
                if Y[j, 0].item() != target:
                    non_target_sample[num_non_target] = X[j]
                    num_non_target += 1
                if num_non_target > poisoned_image_num:
                    break
            if num_non_target > poisoned_image_num:
                break
        return non_target_sample

    def retrain(self,
                train_opt=None,
                lr_scheduler=None,
                retrain_epoch: int = None,
                target: int = None,
                validate_interval: int = None,
                parallel=None,
                poisoned_image_num: int = None,
                epsilon: int = None,
                preprocess_layer=None,
                poison_generation_iteration: int = None,
                decay=None,
                decay_ratio=None,
                decay_iteration=None):
        """
        Retrain the model with normal images and poisoned images whose label haven't be modified, finetune self.model in this process
        :param train_opt: specify whether finetune the whole model or only part
        :param lr_scheduler: specify whether to adjust learning rate 
        :param retrain_epoch: how many epoches that the model needs to retrained 
        :param target: the target class
        :param validate_interval: the interval epoch needed to validate the model
        :param parallel: specify whether to parallel process data
        :param poisoned_image_num: the number of poisoned images
        :param epsilon: the threshold in pixel space to ensure the poisoned image is not visually distinguishable from the target image
        :param preprocess_layer: the chosen specific layer that on which the feature space of source images patched by trigger is close to poisoned images 
        :param poison_generation_iteration: the iteration times used to generate one poison image
        :param decay: specify whether the learning rate decays with iteraion times
        :param decay_ratio: specify the learning rate decay proportion
        :param decay_iteration: specify the number of iteration time interval, the learning rate will decays once
        :returns the model after fine-tuning
        """
        if train_opt is None:
            train_opt = self.train_opt
        if lr_scheduler is None:
            lr_scheduler = self.lr_scheduler
        if retrain_epoch is None:
            retrain_epoch = self.retrain_epoch
        if target is None:
            target = self.target_class
        if validate_interval is None:
            validate_interval = self.validate_interval

        if parallel is None:
            parallel = self.parallel
        if poisoned_image_num is None:
            poisoned_image_num = self.poisoned_image_num
        if epsilon is None:
            epsilon = self.epsilon
        if preprocess_layer is None:
            preprocess_layer = self.preprocess_layer
        if poison_generation_iteration is None:
            poison_generation_iteration = self.poison_generation_iteration
        if decay is None:
            decay = self.decay
        if decay_ratio is None:
            decay_ratio = self.decay_ratio
        if decay_iteration is None:
            decay_iteration = self.decay_iteration

        optimizer = self.model.define_optimizer(train_opt=train_opt,
                                                lr_scheduler=lr_scheduler,
                                                **kwargs)
        optimizer.zero_grad()
        _lr_scheduler = None
        if lr_scheduler:
            _lr_scheduler = optimizer
            optimizer = _lr_scheduler.optimizer
        self.model.train()

        if torch.cuda.is_available() and self.use_gpu == True:
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        _lambda = 0.2 if self.model.dataset.name == 'gtsrb' else 0.6

        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        for _epoch in range(retrain_epoch):
            entropy = 0.0
            counter = 0
            losses.reset()
            top1.reset()
            top5.reset()
            for i, data in enumerate(tqdm(self.model.dataset.loader['train'])):

                X, Y = self.model.get_data(data, mode='train')
                loss = to_tensor(torch.Tensor([0.0]))
                if i == 0:
                    target_image = self.sample_target(target,
                                                      poisoned_image_num)
                    non_target_image = self.sample_non_target(
                        target, poisoned_image_num)
                    poisoned_image = self.generate_poisoned_image(
                        non_target_image,
                        target_image,
                        epsilon=epsilon,
                        preprocess_layer=preprocess_layer,
                        poison_generation_iteration=poison_generation_iteration,
                        decay=decay,
                        decay_ratio=decay_ratio,
                        decay_iteration=decay_iteration
                    )  # ensure only injecting poisoned images once to train set in one epoch
                    batch_target = repeat_to_batch(
                        to_tensor(target).squeeze(), self.poisoned_image_num)
                    X = torch.cat((X, poisoned_image))
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

            print(('Epoch: [%d/%d],' %
                   (_epoch + 1, retrain_epoch)).ljust(25, ' ') +
                  'Loss: %.4f,\tTop1 Acc: %.3f,\tTop5 Acc: %.3f' %
                  (losses.avg, top1.avg, top5.avg))
            if lr_scheduler:
                _lr_scheduler.step()

            if validate_interval != 0:
                if (_epoch + 1
                    ) % validate_interval == 0 or _epoch == retrain_epoch - 1:
                    _, cur_acc, _ = self.model._validate()
                    self.model.train()
                    print(
                        '---------------------------------------------------')
        self.model.zero_grad()
        self.model.eval()
