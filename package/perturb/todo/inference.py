# -*- coding: utf-8 -*-
from .pgd import *

from copy import deepcopy
import math


class Inference(PGD):

    def __init__(self, gradient_method='nes', query_num=100, sigma=1e-3,
                 attack_adapt=False, fake_percent=0.4, dist=5.0,
                 defend_adapt=False, k=1, b=4e-3,
                 active=False, active_percent=0.1, active_multiplier=1, **kwargs):
        super(Inference, self).__init__(mode='black', sigma=sigma, **kwargs)

        self.gradient_method = gradient_method
        self.query_num = query_num

        self.attack_adapt = attack_adapt
        self.fake_percent = fake_percent
        if not self.attack_adapt:
            self.fake_percent = 0.0
        self.dist = dist

        self.defend_adapt = defend_adapt
        self.k = k
        self.b = b

        self.active = active
        self.active_percent = active_percent
        self.active_multiplier = active_multiplier

        self.hess = to_tensor(torch.zeros(1))
        self.hess_b = self.query_num
        self.hess_p = 1
        self.hess_lambda = 1

        print(self.query_num)
        print(self.fake_percent)
        self.fake_query_num = int(self.query_num * self.fake_percent + 0.001)
        self.true_query_num = self.query_num - self.fake_query_num

        self.active_multiplier = 2 ** (1-self.active_multiplier)
        self.attack_grad_list = None

        print('Inference parameter list: ')
        d = self.__dict__
        print(
            {key: d[key] for key in d if '__' not in key and 'hess' not in key and 'temp' not in key})

    @staticmethod
    def phi(x, c=1.35):
        return torch.clamp(x, min=-c, max=c)

    @staticmethod
    def phi_log(x):
        return 1-2/(torch.exp(x)+1)

    def gen_seq(self, X, n, sigma=None, method='nes', cur_iter=0, f=None):
        if sigma is None:
            sigma = self.sigma
        seq = [deepcopy(X.detach())]
        if method == 'nes':
            for i in range(n//2):
                noise = to_tensor(torch.normal(mean=0.0, std=1.0, size=X.shape))
                X1 = X + sigma * noise
                X2 = X - sigma * noise
                seq.append(X1.detach())
                seq.append(X2.detach())
            if n % 2 == 1:
                seq.append(X)
        elif method == 'sgd':
            for i in range(n):
                noise = to_tensor(torch.normal(mean=0.0, std=1.0, size=X.shape))
                X1 = X + sigma * noise
                seq.append(X1.detach())
        elif method == 'hess':
            if cur_iter % self.hess_p == 0:
                self.hess = self.calc_hess(
                    f, X, sigma=sigma, hess_b=self.hess_b, hess_lambda=self.hess_lambda)
                self.hess /= self.hess.norm(p=2)
            for i in range(n):
                noise = to_tensor(torch.normal(mean=0.0, std=1.0, size=X.shape)).view(-1, 1)
                X1 = X + sigma * self.hess.mm(noise).view(X.shape)
                seq.append(X1.detach())
        elif method == 'zoo':
            raise NotImplementedError()
        else:
            print('Current method: ', method)
            raise ValueError(
                'Argument \"method\" should be \"nes\", \"sgd\" or \"hess\"!')
        return seq

    @staticmethod
    def calc_grad(f, X, seq, n, sigma, method='nes'):
        with torch.no_grad():
            g = to_tensor(torch.zeros_like(X))
            if method == 'nes':
                for x in seq:
                    g += f(x).detach()*(x-X)
            elif method == 'sgd' or method == 'hess':
                for x in seq:
                    g += (f(x)-f(X)).detach()*(x-X)
            g /= n*sigma*sigma
        return g.detach()

    # def cal_gradient(self, f, X, n=100, seq=None, method='nes', cur_iter=0):
    #     with torch.no_grad():
    #         X = to_tensor(X)
    #         if method == 'zoo':
    #             return
    #         else:
    #             g = to_tensor(torch.zeros_like(X))
    #             if method == 'nes':
    #                 for i in range(int(n//2)):
    #                     noise = self.gauss_noise(X.shape)
    #                     X1 = X + self.sigma * noise
    #                     X2 = X - self.sigma * noise
    #                     g += f(X1).detach() * noise
    #                     g -= f(X2).detach() * noise
    #                     if seq is not None:
    #                         seq.append(X1)
    #                         seq.append(X2)
    #                 if n % 2 == 1:
    #                     seq.append(X)
    #             elif method == 'sgd':
    #                 for i in range(n):
    #                     noise = self.gauss_noise(X.shape)
    #                     X1 = X + self.sigma * noise
    #                     g += (f(X1)-f(X)).detach() * noise
    #                     if seq is not None:
    #                         seq.append(X1)
    #             elif method == 'hess':
    #                 if cur_iter % self.hess_p == 0:
    #                     self.hess = self.calc_hess(f, X)
    #                     self.hess /= self.hess.norm(p=2)
    #                 for i in range(n):
    #                     noise = self.gauss_noise(X.shape)
    #                     X1 = (X + self.sigma * self.hess.mm(noise).view(X.shape))
    #                     g += (f(X1)-f(X)).detach() * \
    #                         self.hess.mm(noise).view(X.shape)
    #                     if seq is not None:
    #                         seq.append(X1)
    #             g /= n * self.sigma
    #             return g

    @staticmethod
    def calc_hess(f, X, sigma, hess_b, hess_lambda=1, method='gauss'):
        with torch.no_grad():
            if method == 'gauss':
                hess = to_tensor(torch.zeros(len(X.view(-1)), len(X.view(-1))))
                for i in range(hess_b):
                    noise = to_tensor(torch.normal(mean=0.0, std=1.0, size=X.shape))
                    X1 = X + sigma * noise
                    X2 = X - sigma * noise
                    hess += abs(f(X1)+f(X2)-2*f(X)) * \
                        noise.view(-1, 1).mm(noise.view(1, -1))
                hess /= (2*hess_b * sigma*sigma)
                hess += hess_lambda * to_tensor(
                    torch.eye(len(X.view(-1))))
                result = hess.cholesky_inverse()

                # C=[]
                # for i in range(self.hess_b):
                #     noise = to_tensor(m.sample(), dtype='float').view(X.shape)
                #     X1 = X + self.sigma * noise
                #     X2 = X - self.sigma * noise
                #     cur_result = abs(f(X1)+f(X2)-2*f(X)) / (2*self.hess_b * self.sigma*self.sigma)
                #     C.append(noise.view(-1)*math.sqrt(cur_result))

                # C = to_tensor(C).t()
                # (U, S, V) = C.svd()
                # hess_lambda = self.hess_lambda * C.mm(C.t()).norm()
                # # hess += hess_lambda * torch.eye(len(X.view(-1)))
                # result = U.mm((S*S+hess_lambda*to_tensor(torch.eye(len(X.view(-1)))))**(-0.5)-to_tensor(torch.eye(len(
                #     X.view(-1))))/math.sqrt(hess_lambda)).mm(V)+to_tensor(torch.eye(len(X.view(-1))))/math.sqrt(hess_lambda)

            elif method == 'diag':
                raise NotImplementedError()
        return result

    @staticmethod
    def fake_noise(sub_seq, fake_query_num, sigma, dist=1.0):
        X = sub_seq[0].detach()

        noise = to_tensor(torch.normal(mean=0.0, std=1.0, size=X.shape))
        X = X+dist*sigma*noise
        for i in range(fake_query_num):
            sub_seq.append(X)
        return sub_seq

    def get_seq(self, _input, target):
        seq = []

        X_var = deepcopy(_input)

        noise = to_tensor(torch.zeros_like(X_var))

        def loss_func(_X):
            return self.loss(_X, target=target, parallel=False)

        for _iter in range(self.iteration):
            # Attacker generate sequence
            sub_seq = self.gen_seq(X_var, n=self.true_query_num,
                                   sigma=self.sigma, method=self.gradient_method, cur_iter=_iter, f=loss_func)

            # Attack Adaptive
            if self.attack_adapt:
                sub_seq = self.fake_noise(sub_seq, fake_query_num=self.fake_query_num,
                                          sigma=self.sigma, dist=self.dist)
            seq.append(sub_seq)

            # Defense Active
            if self.active:
                cluster = torch.stack(sub_seq)
                est_center = self.get_center(cluster)
                center = est_center
                # center_idx = to_tensor([(x-est_center).norm(p=2)
                #                         for x in sub_seq]).argmin().item()
                # center = sub_seq[center_idx]

                center.requires_grad = True
                self.model.zero_grad()
                center_loss = loss_func(center)
                center_loss.backward()
                real_grad = to_tensor(deepcopy(center.grad))
                real_grad /= real_grad.norm(p=2)
                center.requires_grad = False

                # real_grad.sign_()

                noise_grad = to_tensor(torch.zeros_like(real_grad)).view(-1)
                for multiplier in range(len(noise_grad)//self.model.num_classes):
                    noise_grad[multiplier*self.model.num_classes +
                               (target+_iter) % self.model.num_classes] = 1
                noise_grad = noise_grad.view(real_grad.shape)
                active_grad = self.active_percent*noise_grad + \
                    (1-self.active_percent)*real_grad

                def active_loss(_X):
                    loss = loss_func(center)+self.active_multiplier * \
                        ((_X-center)*active_grad).sum()
                    return loss
                grad = self.calc_grad(f=active_loss, X=sub_seq[0], seq=sub_seq[1:self.true_query_num+1], n=self.true_query_num,
                                      sigma=self.sigma, method=self.gradient_method)/self.active_multiplier
            else:
                grad = self.calc_grad(f=loss_func, X=sub_seq[0], seq=sub_seq[1:self.true_query_num+1], n=self.true_query_num,
                                      sigma=self.sigma, method=self.gradient_method)
            grad.sign_()
            if self.output > 0:
                self.attack_grad_list.append(grad.detach())

            noise = to_valid_img(noise - self.alpha *
                                 grad, - self.epsilon, self.epsilon)
            X_var = to_valid_img((_input + noise).detach())
        return seq

    def get_center(self, cluster):
        if self.defend_adapt:
            T = cluster.median(dim=0)[0]
            S = cluster-repeat_to_batch(T, batch_size=self.batch_size)
            S = 1.4826 * self.b * \
                to_tensor([point.norm(p=2) for point in S]).median().item()
            for i in range(self.k):
                T = T + S/0.4132 * self.phi_log(
                    (cluster-repeat_to_batch(T, batch_size=self.batch_size))/S).median(dim=0)[0]
            return to_tensor(T.detach())
        else:
            return to_tensor(cluster.mean(dim=0)[0].detach())

    def get_bias(self, cluster):
        # if self.defend_adapt:
        T = cluster.median(dim=0)[0]
        S = (cluster-repeat_to_batch(T, batch_size=self.batch_size))
        S = 1.4826 * self.b * to_tensor([point.norm(p=2)
                                         for point in S]).median().item()  # 1.4826
        # B0 = math.sqrt(2)*torch.erfinv(to_tensor(1 / (1-self.fake_percent)-1)).item()
        B0 = 0.0
        B = B0
        for i in range(self.k):
            B = B+S/0.4132 * (self.fake_percent+(1-self.fake_percent)*self.phi_log((cluster -
                                                                                    B)/S).median(dim=0)[0].norm(p=2).item())
        return B
        # else:
        #     return 1.96 * self.sigma/math.sqrt(self.query_num)

    def get_center_bias(self, seq):
        seq_centers = []
        seq_bias = []

        for cluster in seq:
            cluster = torch.stack(cluster)

            T = self.get_center(cluster)
            seq_centers.append(T)

            B = self.get_bias(cluster)
            seq_bias.append(B)

        return seq_centers, seq_bias

    def get_candidate_centers(self, seq, seq_centers, seq_bias):
        center_seq = []
        for i in range(len(seq)):
            center_seq.append([seq_centers[i]])
            # sub_seq = []

            # # idx = to_tensor([(x-seq_centers[i]).norm(p=2)
            # #                  for x in seq[i]]).argmin().item()
            # # sub_seq.append(seq[i][idx])
            # for point in seq[i]:
            #     sub_seq.append(point)
            # norms = to_tensor([(x-seq_centers[i]).norm(p=2) for x in sub_seq])
            # if norms.shape != torch.Size([0]):
            #     idx = norms.argmin()
            #     center_seq.append([sub_seq[idx]])
            # else:
            #     center_seq.append([])
        return center_seq

    def get_detect_result(self, candidate_centers, seq_centers, seq, target=None):
        pair_seq = to_tensor(-torch.ones(self.iteration-1), dtype='long')
        detect_prob = torch.ones(self.model.num_classes)/self.model.num_classes
        for i in range(len(candidate_centers)-1):
            sub_pair_seq = []
            for point in candidate_centers[i]:
                X_var = point.detach()
                X_var.requires_grad = True
                dist_list = torch.zeros(self.model.num_classes)

                for _class in range(self.model.num_classes):
                    loss = self.loss(X_var,
                                     target=to_tensor([_class]), parallel=False)
                    loss.backward()
                    grad = to_tensor(deepcopy(X_var.grad.detach()))
                    grad /= grad.norm(p=2)
                    X_var.grad.zero_()
                    if self.active:
                        noise_grad = to_tensor(
                            torch.zeros_like(X_var)).view(-1)
                        for multiplier in range(int(len(noise_grad)/self.model.num_classes)):
                            noise_grad[multiplier*self.model.num_classes +
                                       (_class+i) % self.model.num_classes] = 1
                        noise_grad = noise_grad.view(X_var.shape)
                        noise_grad /= noise_grad.norm(p=2)
                        grad = self.active_percent*noise_grad + \
                            (1-self.active_percent)*grad
                    # grad.sign_()
                    vec = seq_centers[i+1]-point
                    dist = self.cos_sim(-grad, vec)
                    dist_list[_class] = dist
                    if self.output > 1 and _class == target:
                        print(vec.norm())
                        print(grad.norm())
                        print(self.attack_grad_list[i].norm())
                        print('sim <vec, real>: ', self.cos_sim(vec, -grad))
                        print('sim <est, real>: ',
                              self.cos_sim(self.attack_grad_list[i], grad))
                        print('sim <vec, est>: ',
                              self.cos_sim(vec, -self.attack_grad_list[i]))
                detect_prob = F.softmax(torch.log((2/(1-dist_list)).sub(1)))
                # detect_prob.div_(detect_prob.norm(p=2))
            pair_seq[i] = (detect_prob.argmax())
        return pair_seq.tolist()

    # def get_center_class_pairs(self, candidate_centers, seq_centers, seq, _model, model=None):
    #     pair_seq = []
    #     for i in range(len(candidate_centers)-1):
    #         sub_pair_seq = []
    #         for point in candidate_centers[i]:
    #             # if self.active:
    #             #     vec = seq_centers[i+1]-point
    #             #     _result = vec.view(-1)
    #             #     for j in range(len(_result)):
    #             #         if _result[j] < 0 and j > i:
    #             #             sub_pair_seq.append((j-i) % self.num_classes)

    #                 # print(vec.view(-1)[:self.num_classes])
    #             X_var = deepcopy(point.detach())
    #             X_var.requires_grad = True
    #             dist_list = torch.zeros(self.num_classes)
    #             # print('bound: ', estimate_error + shift_dist)
    #             for _class in range(self.num_classes):
    #                 loss = _model.criterion(model(X_var), to_tensor([_class]))
    #                 loss.backward()
    #                 grad = to_tensor(deepcopy(X_var.grad.detach()))
    #                 X_var.grad.zero_()
    #                 grad.sign_()
    #                 if self.active:
    #                     noise_grad = to_tensor(torch.zeros_like(grad)).view(-1)
    #                     for multiplier in range(int(len(noise_grad)/self.num_classes)):
    #                         noise_grad[multiplier*self.num_classes +
    #                                    (_class+i) % self.num_classes] = 1
    #                     noise_grad = noise_grad.view(grad.shape)
    #                     grad = self.active_percent*noise_grad + \
    #                         (1-self.active_percent)*grad
    #                     grad.sign_()
    #                 vec = seq_centers[i+1]-point
    #                 dist = self.cos_sim(-grad, vec)
    #                 dist_list[_class] = dist
    #             sub_pair_seq.append(dist_list.argmax().item())
    #         pair_seq.append(sub_pair_seq)
    #     return pair_seq

    def inference(self, _input, target, seed=None):

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        if self.output > 0:
            self.output_result(_input=_input, target=target, targeted=True)

        self.attack_grad_list = []

        seq = self.get_seq(_input, target)

        seq_centers, seq_bias = self.get_center_bias(seq)

        # seq_centers = [sub_seq[0] for sub_seq in seq]

        if self.output > 0:
            mean_error = to_tensor([torch.norm(seq_centers[i]-seq[i][0], p=2).item()
                                    for i in range(len(seq))])
            seq_bias = to_tensor(seq_bias)
            print('  Mean Shift Distance: avg %f, min %f , max %f' %
                  (mean_error.mean(), mean_error.min(), mean_error.max()))
            print('  Bias Estimation: avg %f, min %f , max %f' %
                  (seq_bias.mean(), seq_bias.min(), seq_bias.max()))

        candidate_centers = self.get_candidate_centers(
            seq, seq_centers, seq_bias)
        detect_result = self.get_detect_result(
            candidate_centers, seq_centers, seq, target=target)

        attack_result = []
        for l in range(self.iteration-1):
            attack_result.append(self.model(
                seq[l+1][0], parallel=False).squeeze().argmax())
        attack_result = torch.stack(attack_result)

        attack_succ = self.iteration
        detect_succ = self.iteration

        detect_true = True
        for i in range(self.iteration-1):
            if attack_result[i] == target and attack_result[min(i+1, self.iteration-2)] == target and attack_succ == self.iteration:
                attack_succ = i
            if detect_result[i] == detect_result[min(i+1, self.iteration-2)] and detect_succ == self.iteration and detect_true:
                if detect_result[i] == target:
                    detect_succ = i
                else:
                    detect_true = False
        if self.output > 0:
            # print('candidate centers: ', [len(i) for i in candidate_centers])
            print('target class: ', target)
            print('detect: ', detect_result)
            print('attack: ', attack_result)
            print(attack_succ)
            print(detect_succ)
            print()
        result = []
        for i in range(self.iteration-1):
            result.append('draw')
        if attack_succ < detect_succ:
            for i in range(attack_succ, self.iteration-1):
                result[i] = 'lose'
        elif attack_succ > detect_succ:
            for i in range(detect_succ, self.iteration-1):
                result[i] = 'win'
        elif attack_succ == detect_succ:
            pass
        else:
            raise ValueError()
        return result, to_tensor(detect_result), to_tensor(attack_result), attack_succ, detect_succ
