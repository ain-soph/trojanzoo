#!/usr/bin/env python3

from trojanvision.attacks import PGD
from trojanvision.environ import env
from trojanzoo.defenses import Defense
from trojanzoo.utils.tensor import repeat_to_batch
from trojanzoo.utils.output import prints

import torch
import torch.nn as nn
import torch.autograd
import torch.nn.functional as F
import argparse


class AdvMind(Defense):

    name: str = 'advmind'

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--attack_adapt', action='store_true', help='Adaptive attack to add fake queries.')
        group.add_argument('--fake_percent', type=float, help='fake query percentage.')
        group.add_argument('--dist', type=float, help='fake query noise std.')
        group.add_argument('--defend_adapt', action='store_true', help='Robust location M-estimator.')
        group.add_argument('--active', action='store_true', help='Proactive solicitation.')
        group.add_argument('--active_percent', type=float, help='Active gradient weight.')
        return group

    def __init__(self, attack_adapt: bool = False, fake_percent: float = 0.3, dist: float = 50.0,
                 defend_adapt: bool = False, k: int = 1, b: float = 4e-3,
                 active: bool = False, active_percent: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.attack: PGD = self.attack

        self.param_list['advmind'] = []
        if attack_adapt:
            self.param_list['advmind'].extend(['fake_percent', 'dist', 'active_percent'])
        if defend_adapt:
            self.param_list['advmind'].extend(['k', 'b'])
        if active:
            self.param_list['advmind'].extend(['active_percent'])

        self.attack_adapt: bool = attack_adapt
        self.fake_percent: float = fake_percent
        if not self.attack_adapt:
            self.fake_percent: float = 0.0
        self.dist: float = dist

        self.defend_adapt: bool = defend_adapt
        self.k: int = k
        self.b: float = b

        self.active: bool = active
        self.active_percent: float = active_percent

        # Temp Variable
        self.fake_query_num: int = int(self.attack.query_num * self.fake_percent)
        self.true_query_num: int = self.attack.query_num - self.fake_query_num
        self.attack_grad_list: list[torch.Tensor] = []

        self.cos_sim = nn.CosineSimilarity()

    def detect(self):
        zeros = [0] * (self.attack.iteration - 1)
        result_dict = {'draw': zeros.copy(), 'win': zeros.copy(), 'lose': zeros.copy()}
        counter = 0
        attack_succ_num = 0.
        detect_succ_num = 0.

        validset = self.dataset.get_dataset('valid')
        testset, _ = self.dataset.split_dataset(validset, percent=0.3)
        loader = self.dataset.get_dataloader(mode='valid', dataset=testset,
                                             batch_size=1, shuffle=True)
        for i, data in enumerate(loader):
            if counter >= 200:
                break
            print('img idx: ', i)
            _input, _label = self.model.remove_misclassify(data)
            if len(_label) == 0:
                print('misclassification')
                print()
                print('------------------------------------')
                continue
            target = self.model.generate_target(_input)
            result, _, _, attack_succ, detect_succ = self.inference(_input, target)
            for _iter in range(len(result)):
                result_dict[result[_iter]][_iter] += 1
            counter += 1
            if attack_succ != self.attack.iteration:
                attack_succ_num += 1
            if detect_succ != self.attack.iteration:
                detect_succ_num += 1
            attack_succ_rate = attack_succ_num / counter
            detect_succ_rate = detect_succ_num / counter
            print('draw: ', result_dict['draw'])
            print('win : ', result_dict['win'])
            print('lose: ', result_dict['lose'])
            print()
            print('total: ', counter)
            print('attack succ rate: ', attack_succ_rate)
            print('detect succ rate: ', detect_succ_rate)
            print('------------------------------------')

    def inference(self, _input: torch.Tensor, target: torch.Tensor):
        # ------------------------------- Init --------------------------------- #
        torch.manual_seed(env['seed'])
        if 'start' in self.output:
            self.attack.output_info(_input=_input, noise=torch.zeros_like(_input), target=target,
                                    loss_fn=lambda _X: self.model.loss(_X, target))
        self.attack_grad_list: list[torch.Tensor] = []
        # ------------------------ Attacker Seq -------------------------------- #
        seq = self.get_seq(_input, target)  # Attacker cluster sequences (iter, query_num+1, C, H, W)
        seq_centers, seq_bias = self.get_center_bias(seq)  # Defender cluster center estimate
        # seq_centers: (iter, 1, C, H, W)   seq_bias: (iter)
        # seq_centers = seq[:, 0]  # debug
        if 'start' in self.output:
            mean_error = (seq_centers[:, 0] - seq[:, 0]).abs().flatten(start_dim=1).amax(dim=1)
            print('Mean Shift Distance: '.ljust(25) +
                  f'avg {mean_error.mean():<10.5f} min {mean_error.min():<10.5f} max {mean_error.max():<10.5f}')
            print('Bias Estimation: '.ljust(25) +
                  f'avg {seq_bias.mean():<10.5f} min {seq_bias.min():<10.5f} max {seq_bias.max():<10.5f}')
        # candidate_centers = self.get_candidate_centers(seq, seq_centers, seq_bias)  # abandoned
        # candidate_centers = seq_centers
        detect_result = self.get_detect_result(seq_centers, target=target)
        attack_result = self.model(seq[:, 0]).argmax(dim=1).detach().cpu()

        attack_succ = self.attack.iteration
        detect_succ = self.attack.iteration

        detect_true = True
        for i in range(self.attack.iteration - 1):
            if attack_result[i] == target and \
                    attack_result[min(i + 1, self.attack.iteration - 2)] == target and \
                    attack_succ == self.attack.iteration:
                attack_succ = i
            if (detect_true and detect_result[i] == detect_result[min(i + 1, self.attack.iteration - 2)]
                    and detect_succ == self.attack.iteration):
                if detect_result[i] == target:
                    detect_succ = i
                else:
                    detect_true = False
        if 'end' in self.output:
            # print('candidate centers: ', [len(i) for i in candidate_centers])
            print('Detect Iter: ', detect_succ)
            prints(detect_result.tolist(), indent=12)
            print('Attack Iter: ', attack_succ)
            prints(attack_result.tolist(), indent=12)
            print()
        result = ['draw'] * (self.attack.iteration - 1)
        if attack_succ < detect_succ:
            for i in range(attack_succ, self.attack.iteration - 1):
                result[i] = 'lose'
        elif attack_succ > detect_succ:
            for i in range(detect_succ, self.attack.iteration - 1):
                result[i] = 'win'
        elif attack_succ == detect_succ:
            pass
        else:
            raise ValueError()
        return result, detect_result, attack_result, attack_succ, detect_succ

    def get_seq(self, _input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        seq = []
        X = _input.clone()
        noise: torch.Tensor = torch.zeros_like(_input)
        if isinstance(target, int):
            target = target * torch.ones(len(_input), dtype=torch.long, device=_input.device)

        def loss_fn(_X: torch.Tensor, **kwargs):
            t = target
            if len(_X) != len(target) and len(target) == 1:
                t = target * torch.ones(len(_X), dtype=torch.long, device=_X.device)
            return F.cross_entropy(self.model(_X), t, **kwargs)

        for _iter in range(self.attack.iteration):
            # Attacker generate sequence
            if self.attack.grad_method == 'hess' and _iter % self.hess_p == 0:  # TODO
                self.hess = self.calc_hess(loss_fn, X, sigma=self.attack.sigma,
                                           hess_b=self.hess_b, hess_lambda=self.hess_lambda)
                self.hess /= self.hess.norm(p=2)
            cluster = self.attack.gen_seq(X, query_num=self.true_query_num)[:, 0]  # (query_num+1, C, H, W)
            # Attack Adaptive
            if self.attack_adapt:
                fake_seq = self.get_fake_seq(X=X)
                cluster = torch.cat([cluster, fake_seq])
            seq.append(cluster)
            # Defense Active
            if self.active:
                center = self.get_center(cluster)
                # center_idx = (cluster - center).flatten(start_dim=1).norm(p=2, dim=1).argmin()
                # center = cluster[center_idx]
                center.requires_grad_()
                loss = loss_fn(center)
                real_grad = torch.autograd.grad(loss, center)[0]
                center.requires_grad = False
                real_grad /= float(real_grad.abs().max())
                # real_grad.sign_()
                noise_grad = torch.zeros_like(real_grad).flatten()
                offset = (target + _iter) % self.model.num_classes
                for multiplier in range(len(noise_grad) // self.model.num_classes):
                    noise_grad[multiplier * self.model.num_classes + offset] = 1
                noise_grad = noise_grad.view(real_grad.shape)
                active_grad = self.active_percent * noise_grad + \
                    (1 - self.active_percent) * real_grad

                def active_loss(_X: torch.Tensor, **kwargs):
                    org_loss = loss_fn(center, **kwargs)
                    active_loss = ((_X - center) * active_grad).flatten(start_dim=1).sum(dim=1)
                    return org_loss + active_loss
                grad = self.attack.calc_seq(f=active_loss, seq=cluster[:self.true_query_num + 1])
            else:
                grad = self.attack.calc_seq(f=loss_fn, seq=cluster[:self.true_query_num + 1])
            grad.sign_()
            if 'middle' in self.output:
                self.attack_grad_list.append(grad.clone())

            noise = (noise - self.attack.pgd_alpha * grad).clamp(-self.attack.pgd_eps, self.attack.pgd_eps)
            X = (_input + noise).clamp(0, 1)
            noise = X - _input
        return torch.stack(seq)  # (iter, query_num+1, C, H, W)

    def get_fake_seq(self, X: torch.Tensor) -> torch.Tensor:
        if X.dim() == 4:
            X = X[0]
        noise = torch.normal(mean=0.0, std=1.0, size=X.shape, device=X.device)
        fake_seq = X + self.dist * self.attack.sigma * noise
        return repeat_to_batch(fake_seq, batch_size=self.fake_query_num)

    # cluster: (query_num, C, H, W)
    def get_center(self, cluster: torch.Tensor) -> torch.Tensor:
        if self.defend_adapt:
            T = cluster.median(dim=0)[0].unsqueeze(0)   # (1, C, H, W)
            S = cluster - T  # (query_num, C, H, W)
            S = S.flatten(start_dim=1)  # (query_num, C*H*W)
            S = 1.4826 * self.b * torch.median(S.norm(p=2, dim=1))   # scalar
            for i in range(self.k):
                T = T + S / 0.4132 * self.phi_log(
                    (cluster - T) / S).median(dim=0)[0].unsqueeze(0)   # (1, C, H, W)
            return T
        else:
            return cluster.mean(dim=0).unsqueeze(0)  # (1, C, H, W)

    def get_bias(self, cluster: torch.Tensor) -> float:
        T = cluster.median(dim=0)[0].unsqueeze(0)
        S = cluster - T
        S = S.flatten(start_dim=1)
        S = 1.4826 * self.b * torch.median(S.norm(p=2, dim=1))
        # B = math.sqrt(2)*torch.erfinv(torch.as_tensor([1 / (1-self.fake_percent)-1])).item()
        B = 0.0
        for i in range(self.k):
            result = float(self.phi_log((cluster - B) / S).median(dim=0)[0].norm(p=2))
            B = B + S / 0.4132 * (self.fake_percent + (1 - self.fake_percent) * result)
        return B

    def get_center_bias(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_centers = []
        seq_bias = []
        for cluster in seq:
            T = self.get_center(cluster)
            seq_centers.append(T)
            B = self.get_bias(cluster)
            seq_bias.append(B)
        return torch.stack(seq_centers), torch.stack(seq_bias)

    def get_detect_result(self, seq_centers: torch.Tensor, target=None):
        pair_seq = -torch.ones(self.attack.iteration - 1, dtype=torch.long)
        detect_prob = torch.ones(self.model.num_classes) / self.model.num_classes
        for i in range(len(seq_centers) - 1):
            X: torch.Tensor = seq_centers[i].clone()
            dist_list = torch.zeros(self.model.num_classes)

            for _class in range(self.model.num_classes):
                _label = _class * torch.ones(len(X), dtype=torch.long, device=X.device)
                X.requires_grad_()
                loss = self.model.loss(X, _label)
                grad: torch.Tensor = torch.autograd.grad(loss, X)[0]
                X.requires_grad = False
                grad /= grad.abs().max()
                if self.active:
                    noise_grad = torch.zeros(X.numel(), device=X.device)
                    offset = (_class + i) % self.model.num_classes

                    for multiplier in range(len(noise_grad) // self.model.num_classes):
                        noise_grad[multiplier * self.model.num_classes + offset] = 1
                    noise_grad = noise_grad.view(X.shape)
                    # noise_grad /= noise_grad.abs().max()
                    grad = self.active_percent * noise_grad + \
                        (1 - self.active_percent) * grad
                grad.sign_()
                vec = seq_centers[i + 1] - X
                dist = self.cos_sim(-grad.flatten(), vec.flatten())
                dist_list[_class] = dist
                if 'middle' in self.output and _class == target:
                    print('sim <vec, real>: ', self.cos_sim(vec.flatten(), -grad.flatten()))
                    print('sim <est, real>: ',
                          self.cos_sim(self.attack_grad_list[i].flatten(), grad.flatten()))
                    print('sim <vec, est>: ',
                          self.cos_sim(vec.flatten(), -self.attack_grad_list[i].flatten()))
            # todo: Use atanh for normalization after pytorch 1.6
            detect_prob = torch.log((2 / (1 - dist_list)) - 1).softmax(0)
            # detect_prob.div_(detect_prob.norm(p=2))
            pair_seq[i] = detect_prob.argmax().item()
        return pair_seq

    @staticmethod
    def phi_log(x: torch.Tensor) -> torch.Tensor:
        return 1 - 2 / (x.exp() + 1)

    # -------------------------- Unused ----------------------------- #

    @staticmethod
    def phi(x: torch.Tensor, c: float = 1.35) -> torch.Tensor:
        return x.clamp(-c, c)

    # Unused TODO
    @staticmethod
    def get_candidate_centers(seq: torch.Tensor, seq_centers: torch.Tensor, seq_bias: torch.Tensor):
        center_seq = []
        for i in range(len(seq)):
            # for point in seq[i]:
            #     sub_seq.append(point)

            # sub_seq: list[torch.Tensor] = []
            # idx = torch.tensor([(x - seq_centers[i]).norm(p=2).item()
            #                     for x in seq[i]]).argmin().item()
            # sub_seq.append(seq[i][idx])
            # norms = [(x - seq_centers[i]).norm(p=2) for x in sub_seq]
            norms: torch.Tensor = (seq - seq_centers[i]).flatten(1).norm(p=2, dim=-1)
            if norms.dim():
                idx = norms.argmin()
                center_seq.append([seq[idx]])
            else:
                center_seq.append([])
        return center_seq

    # Unused
    def get_center_class_pairs(self, candidate_centers: torch.Tensor, seq_centers: torch.Tensor, seq: torch.Tensor):
        pair_seq = []
        for i in range(len(candidate_centers) - 1):
            sub_pair_seq = []
            for point in candidate_centers[i]:
                # if self.active:
                #     vec = seq_centers[i+1]-point
                #     _result = vec.view(-1)
                #     for j in range(len(_result)):
                #         if _result[j] < 0 and j > i:
                #             sub_pair_seq.append((j-i) % self.num_classes)
                # print(vec.view(-1)[:self.num_classes])
                point: torch.Tensor
                x = point.clone()
                dist_list: torch.Tensor = torch.zeros(self.model.num_classes)
                # print('bound: ', estimate_error + shift_dist)
                for _class in range(self.model.num_classes):
                    x.requires_grad_()
                    loss = self.model.loss(x, _class)
                    grad = torch.autograd.grad(loss, x)[0]
                    x.requires_grad = False
                    grad.sign_()
                    if self.active:
                        noise_grad = torch.zeros_like(grad).flatten()
                        offset = (_class + i) % self.model.num_classes
                        for multiplier in range(int(len(noise_grad) / self.model.num_classes)):
                            noise_grad[multiplier * self.model.num_classes + offset] = 1
                        noise_grad = noise_grad.view(grad.shape)
                        grad = self.active_percent * noise_grad + \
                            (1 - self.active_percent) * grad
                        grad.sign_()
                    vec = seq_centers[i + 1] - point
                    dist = self.cos_sim(-grad.flatten(), vec.flatten())
                    dist_list[_class] = dist
                sub_pair_seq.append(dist_list.argmax().item())
            pair_seq.append(sub_pair_seq)
        return pair_seq
