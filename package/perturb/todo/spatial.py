# -*- coding: utf-8 -*-
from perturb.perturb import *


class Spatial(Perturb):
    def __init__(self, name='Spatial', iteration=20, tau=0.05, lr=0.0001, output=None, **kwargs):
        super(Spatial, self).__init__(name=name, **kwargs)

        self.tau = tau
        self.lr = lr

    def perturb(self, _model, _input, target=None, iteration=None, stop_confidence=0, mode='white', model=None, output=None, tau=None, **kwargs):
        if tau is None:
            tau = self.tau
        k = stop_confidence

        model, criterion, _input, target, untarget, output, kwargs = self.init_perturb(
            _model=_model, _input=_input, target=target, watermark=True, iteration=iteration, mode=mode, model=model, output=output, **kwargs)
        mode = kwargs['mode']
        iteration = kwargs['iteration']

        flows = to_tensor(torch.zeros(
            self.batch_size, 2, _input.shape[-2], _input.shape[-1]))
        flows[:, 0] = 1
        Flow = StadvFlow()
        FlowLoss = StadvFlowLoss()

        flows.requires_grad = True

        optimizer = optim.Adam([flows], self.lr)
        optimizer.zero_grad()

        def loss_func(_flows):
            flow_loss = FlowLoss(_flows)
            _X = Flow(_input.detach(), _flows)
            temp1 = model(_X)
            temp2 = temp1.max(-1)[0] - \
                temp1[to_tensor(range(self.batch_size)), target]
            adv_loss = to_tensor(temp2.clamp(min=k))
            loss = adv_loss + tau * flow_loss
            loss = loss.sum(-1)
            if untarget:
                loss = -loss
            return loss

        for _iter in range(iteration):

            if mode == 'white':
                loss = loss_func(flows)
                loss.backward()
            elif mode == 'black':
                grad = self.cal_gradient(loss_func, flows)
                flows.grad = grad
            else:
                print('Value of Parameter "mode" should be "white" or "black"!')
                sys.exit(-1)
            optimizer.step()
            flows.grad = None

            X_var = Flow(_input, flows)

            _result = F.softmax(model(X_var))
            _confidence, _classification = _result.max(1)
            self.output_middle(target, _result, _iter, output)

            if _classification.equal(target) and _confidence.min() > stop_confidence:
                return X_var.detach(), _iter+1
        return X_var.detach(), None


class StadvTVLoss(nn.Module):

    def forward(self, flows):
        padded_flows = F.pad(flows, (1, 1, 1, 1), mode='replicate')
        height, width = flows.size(2), flows.size(3)
        n = float(np.sqrt(height * width))
        shifted_flows = [
            padded_flows[:, :, 2:, 2:],
            padded_flows[:, :, 2:, :-2],
            padded_flows[:, :, :-2, 2:],
            padded_flows[:, :, :-2, :-2]
        ]

        diffs = [(flows[:, 1] - shifted_flow[:, 1]) ** 2 + (flows[:, 0] - shifted_flow[:, 0]) ** 2
                 for shifted_flow in shifted_flows]
        loss = torch.stack(diffs).sum(2, keepdim=True).sum(
            3, keepdim=True).sum(0, keepdim=True).view(-1)
        loss = torch.sqrt(loss)
        return loss / n


class StadvFlowLoss(nn.Module):

    def forward(self, flows, epsilon=1e-8):
        padded_flows = F.pad(flows, (1, 1, 1, 1), mode='replicate')
        shifted_flows = [
            padded_flows[:, :, 2:, 2:],
            padded_flows[:, :, 2:, :-2],
            padded_flows[:, :, :-2, 2:],
            padded_flows[:, :, :-2, :-2]
        ]

        diffs = [torch.sqrt((flows[:, 1] - shifted_flow[:, 1]) ** 2 +
                            (flows[:, 0] - shifted_flow[:, 0]) ** 2 +
                            epsilon) for shifted_flow in shifted_flows
                 ]
        # shape: (4, n, h - 1, w - 1) => (n, )
        loss = torch.stack(diffs).sum(2, keepdim=True).sum(
            3, keepdim=True).sum(0, keepdim=True).view(-1)
        return loss


class StadvFlow(nn.Module):

    def forward(self, images, flows):
        batch_size, n_channels, height, width = images.shape
        basegrid = torch.stack(torch.meshgrid([torch.arange(height, device=images.device),
                                               torch.arange(width, device=images.device)]))
        batched_basegrid = basegrid.expand(batch_size, -1, -1, -1)

        sampling_grid = batched_basegrid.float() + flows
        sampling_grid_x = torch.clamp(
            sampling_grid[:, 1], 0., float(width) - 1)
        sampling_grid_y = torch.clamp(
            sampling_grid[:, 0], 0., float(height) - 1)

        x0 = sampling_grid_x.floor().long()
        x1 = x0 + 1
        y0 = sampling_grid_y.floor().long()
        y1 = y0 + 1

        x0.clamp_(0, width - 2)
        x1.clamp_(0, width - 1)
        y0.clamp_(0, height - 2)
        y1.clamp_(0, height - 1)

        b = torch.arange(batch_size).view(
            batch_size, 1, 1).expand(-1, height, width)

        Ia = images[b, :, y0, x0].permute(0, 3, 1, 2)
        Ib = images[b, :, y1, x0].permute(0, 3, 1, 2)
        Ic = images[b, :, y0, x1].permute(0, 3, 1, 2)
        Id = images[b, :, y1, x1].permute(0, 3, 1, 2)

        x0 = x0.float()
        x1 = x1.float()
        y0 = y0.float()
        y1 = y1.float()

        wa = (x1 - sampling_grid_x) * (y1 - sampling_grid_y)
        wb = (x1 - sampling_grid_x) * (sampling_grid_y - y0)
        wc = (sampling_grid_x - x0) * (y1 - sampling_grid_y)
        wd = (sampling_grid_x - x0) * (sampling_grid_y - y0)

        wa = wa.unsqueeze(1)
        wb = wb.unsqueeze(1)
        wc = wc.unsqueeze(1)
        wd = wd.unsqueeze(1)

        perturbed_image = torch.stack(
            [wa * Ia, wb * Ib, wc * Ic, wd * Id]).sum(0)
        return perturbed_image
