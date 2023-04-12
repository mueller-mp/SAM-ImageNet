import torch
import torch.nn.functional as F
import random
import contextlib
from torch.distributed import ReduceOp
from torch.nn.modules.batchnorm import _BatchNorm

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05,weight_dropout=0.,adaptive=False, layerwise=False, elementwise_linf=False, nograd_cutoff=0.0,opt_dropout=0.0,temperature=100, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        if layerwise:
            assert adaptive, f"layerwise only possible when ASAM flag is True"
        if elementwise_linf:
            assert adaptive, f"elementwise_linf only possible when ASAM flag is True"

        self.args ={"nograd_cutoff":nograd_cutoff,"opt_dropout":opt_dropout,"temperature":temperature}

        defaults = dict(rho=rho,adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            group["rho"] = rho
            group["adaptive"] = adaptive
            group['layerwise']=layerwise
            group['elementwise_linf']=elementwise_linf
        self.weight_dropout = weight_dropout
        self.paras = None

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        #first order sum 
        taylor_appro = 0
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-7) / (1-self.weight_dropout)
            for p in group["params"]:
                p.requires_grad = True
                if p.grad is None: continue
                #original sam
                # e_w = p.grad * scale.to(p)
                #asam
                if group['elementwise_linf']: # elementwise l-infinity
                    assert group['adaptive']
                    e_w = torch.abs(p) * torch.sign(p.grad) * (group["rho"]+1e-7)
                else:
                    e_w = ((torch.norm(p)**2 if group['layerwise'] else torch.pow(p, 2)) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                    # e_w = (( torch.pow(p, 2)) if group[
                    # "adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
                # p.add_(e_w * 0.1)  old: don't do full step
                # if self.state[p]:
                    # p.sub_(self.state[p]["e_w"])
                self.state[p]["e_w"] = e_w

                taylor_appro += (p.grad**2).sum()

        if zero_grad: self.zero_grad()
        return taylor_appro * scale.to(p)

    @torch.no_grad()
    def first_step_esam(self, zero_grad=False):
        # first order sum
        taylor_appro = 0
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-7) / (1 - self.weight_dropout)
            for p in group["params"]:
                p.requires_grad = True
                if p.grad is None: continue
                # original sam
                # e_w = p.grad * scale.to(p)
                # asam
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w * 0.1)  # climb to the local maximum "w + e(w)"
                # if self.state[p]:
                # p.sub_(self.state[p]["e_w"])
                self.state[p]["e_w"] = e_w

                taylor_appro += (p.grad ** 2).sum()

        if zero_grad: self.zero_grad()
        return taylor_appro * scale.to(p)

    # experimental, not needed
    @torch.no_grad()
    def first_half(self, zero_grad=False):
        #first order sum 
        for group in self.param_groups:
            for p in group["params"]:
                if self.state[p]:
                    p.add_(self.state[p]["e_w"]*0.90)  # climb to the local maximum "w + e(w)"

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or not self.state[p]: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"
                self.state[p]["e_w"] = 0
                # self.state[p] = {}

                if (random.random() > (1-self.weight_dropout)) or not group['sam']:
                    p.requires_grad = False

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    def step(self):
        inputs,targets,loss_fct,model,defined_backward,sam_variant ,pickmiddle = self.paras
        assert defined_backward is not None, "Sharpness Aware Minimization requires defined_backward, but it was not provided"
        args = self.args

        # assert hasattr(model,"require_backward_grad_sync")
        # assert hasattr(model,"require_forward_param_sync")
        if sam_variant=='esam':
            # orginal
            model.require_backward_grad_sync = False
            model.require_forward_param_sync = True

            cutoff = int(len(targets) * args["nograd_cutoff"])
            if cutoff != 0:
                with torch.no_grad():
                    l_before_1 = loss_fct(inputs[:cutoff], targets[:cutoff])

            l_before_2 = loss_fct(inputs[cutoff:], targets[cutoff:])
            loss = l_before_2
            l_before = torch.cat((l_before_1, l_before_2.clone().detach()), 0).detach()
            predictions = None
            return_loss = loss.clone().detach()
            self.returnthings = (predictions, return_loss)
            loss = loss.mean()
            defined_backward(loss)
            self.first_step_esam(True)

            model.require_backward_grad_sync = True
            model.require_forward_param_sync = False

            with torch.no_grad():
                l_after = loss_fct(inputs, targets)
                phase2_coeff = (l_after - l_before) / args["temperature"]
                coeffs = F.softmax(phase2_coeff).detach()

                # codes for sorting
                prob = 1 - args["opt_dropout"]
                if prob >= 0.99:
                    indices = range(len(targets))
                elif not pickmiddle:
                    pp = int(len(coeffs) * prob)
                    cutoff, _ = torch.topk(phase2_coeff, pp)
                    cutoff = cutoff[-1]
                    # cutoff = 0
                    # select top k%
                    indices = [phase2_coeff > cutoff]
                else:
                    floating = 0.1
                    pp_head = int(len(coeffs) * (prob + floating))
                    pp_tail = int(len(coeffs) * (floating))
                    cutoff_head = torch.topk(phase2_coeff, pp_head)[0][-1]
                    cutoff_tail = torch.topk(phase2_coeff, pp_tail)[0][-1]
                    # cutoff = 0
                    # select top k%
                    indices_head = phase2_coeff > cutoff_head
                    indices_tail = phase2_coeff < cutoff_tail
                    indices = torch.logical_and(indices_head, indices_tail)

            # second forward-backward step
            self.first_half()

            # model.require_backward_grad_sync = True
            # model.require_forward_param_sync = False

            loss = loss_fct(inputs[indices], targets[indices])
            loss = (loss * coeffs[indices]).sum()
            defined_backward(loss)
            self.second_step(True)

            # reusume DDP
            model.require_backward_grad_sync = True
            model.require_forward_param_sync = True


            # modified by max
            # model.require_backward_grad_sync = False
            # model.require_forward_param_sync = True
            #
            # cutoff = int(len(targets) * args["nograd_cutoff"])
            # if cutoff != 0:
            #     with torch.no_grad():
            #         l_before_1 = loss_fct(inputs[:cutoff],targets[:cutoff])
            #
            # l_before_2 = loss_fct(inputs[cutoff:],targets[cutoff:])
            # loss = l_before_2
            # l_before = torch.cat((l_before_1,l_before_2.clone().detach()),0).detach()
            # predictions = None
            # return_loss = loss.clone().detach()
            # self.returnthings = (predictions,return_loss)
            # loss = loss.mean()
            # defined_backward(loss)
            # self.first_step(True)
            #
            # model.require_backward_grad_sync = True
            # model.require_forward_param_sync = False
            #
            #
            #
            # with torch.no_grad():
            #     l_after = loss_fct(inputs,targets)
            #     phase2_coeff = (l_after-l_before)/args["temperature"]
            #     coeffs = F.softmax(phase2_coeff).detach()
            #
            #     #codes for sorting
            #     prob = 1 - args["opt_dropout"]
            #     if prob >=0.99:
            #         indices = range(len(targets))
            #     elif not pickmiddle:
            #         pp = int(len(coeffs) * prob)
            #         cutoff,_ = torch.topk(phase2_coeff,pp)
            #         cutoff = cutoff[-1]
            #         # cutoff = 0
            #         #select top k%
            #         indices = [phase2_coeff > cutoff]
            #     else:
            #         floating = 0.1
            #         pp_head = int(len(coeffs) * (prob+floating))
            #         pp_tail = int(len(coeffs) * (floating))
            #         cutoff_head = torch.topk(phase2_coeff,pp_head)[0][-1]
            #         cutoff_tail = torch.topk(phase2_coeff,pp_tail)[0][-1]
            #         # cutoff = 0
            #         #select top k%
            #         indices_head = phase2_coeff > cutoff_head
            #         indices_tail = phase2_coeff < cutoff_tail
            #         indices = torch.logical_and(indices_head,indices_tail)
            #
            #
            # # second forward-backward step
            # # self.first_half()
            #
            # # model.require_backward_grad_sync = True
            # # model.require_forward_param_sync = False
            #
            #
            # loss = loss_fct(inputs[indices], targets[indices])
            # loss = (loss * coeffs[indices]).sum()
            # defined_backward(loss)
            # self.second_step(True)
            #
            # #reusume DDP
            # model.require_backward_grad_sync = True
            # model.require_forward_param_sync = True

        elif sam_variant=='sam':
            self.weight_dropout=0.
            model.require_backward_grad_sync = False
            model.require_forward_param_sync = True

            loss = loss_fct(inputs, targets)
            predictions = None
            return_loss = loss.clone().detach()
            self.returnthings = (predictions, return_loss)
            loss = loss.mean()
            defined_backward(loss)
            self.first_step(True)

            model.require_backward_grad_sync = True
            model.require_forward_param_sync = False

            # second forward-backward step
            # self.first_half()

            # model.require_backward_grad_sync = True
            # model.require_forward_param_sync = False

            loss = loss_fct(inputs, targets)
            loss = (loss).mean()
            defined_backward(loss)
            self.second_step(True)

            # reusume DDP
            model.require_backward_grad_sync = True
            model.require_forward_param_sync = True
 
        elif sam_variant=='base':
            loss = loss_fct(inputs, targets)
            loss = loss.mean()
            defined_backward(loss)
            self.base_optimizer.step()
            self.zero_grad()
            predictions = None
            self.returnthings = (predictions,loss)

        else:
            raise NotImplementedError

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        #original sam 
                        # p.grad.norm(p=2).to(shared_device)
                        #asam 
                        (((torch.norm(p) if group['layerwise'] else torch.abs(p)) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        # (((torch.abs(p)) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if ((p.grad is not None))
                    ]),
                    p=2
               )
        return norm


# GSAM
class GSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, model, gsam_alpha, rho_scheduler=None, rho=1., adaptive=False, perturb_eps=1e-12,
                 grad_reduce='mean', **kwargs):
        defaults = dict(adaptive=adaptive, **kwargs)
        super(GSAM, self).__init__(params, defaults)
        self.model = model
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.adaptive = adaptive
        self.rho_scheduler = rho_scheduler
        self.perturb_eps = perturb_eps
        self.alpha = gsam_alpha
        self.rho_t = rho

        # initialize self.rho_t
        if self.rho_scheduler is not None:
            self.update_rho_t()

        # set up reduction for gradient across workers
        if grad_reduce.lower() == 'mean':
            if hasattr(ReduceOp, 'AVG'):
                self.grad_reduce = ReduceOp.AVG
                self.manual_average = False
            else:  # PyTorch <= 1.11.0 does not have AVG, need to manually average across processes
                self.grad_reduce = ReduceOp.SUM
                self.manual_average = True
        elif grad_reduce.lower() == 'sum':
            self.grad_reduce = ReduceOp.SUM
            self.manual_average = False
        else:
            raise ValueError('"grad_reduce" should be one of ["mean", "sum"].')

    @torch.no_grad()
    def update_rho_t(self):
        self.rho_t = self.rho_scheduler.step()
        return self.rho_t

    @torch.no_grad()
    def perturb_weights(self, rho=0.0):
        grad_norm = self._grad_norm(weight_adaptive=self.adaptive)
        for group in self.param_groups:
            scale = rho / (grad_norm + self.perturb_eps)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_g"] = p.grad.data.clone()
                e_w = p.grad * scale.to(p)
                if self.adaptive:
                    e_w *= torch.pow(p, 2)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]['e_w'] = e_w

    @torch.no_grad()
    def unperturb(self):
        for group in self.param_groups:
            for p in group['params']:
                if 'e_w' in self.state[p].keys():
                    p.data.sub_(self.state[p]['e_w'])

    @torch.no_grad()
    def gradient_decompose(self, alpha=0.0):
        # calculate inner product
        inner_prod = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                inner_prod += torch.sum(
                    self.state[p]['old_g'] * p.grad.data
                )

        # get norm
        new_grad_norm = self._grad_norm()
        old_grad_norm = self._grad_norm(by='old_g')

        # get cosine
        cosine = inner_prod / (new_grad_norm * old_grad_norm + self.perturb_eps)

        # gradient decomposition
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                vertical = self.state[p]['old_g'] - cosine * old_grad_norm * p.grad.data / (
                            new_grad_norm + self.perturb_eps)
                p.grad.data.add_(vertical, alpha=-alpha)

    @torch.no_grad()
    def _sync_grad(self):
        if torch.distributed.is_initialized():  # synchronize final gardients
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    if self.manual_average:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
                        world_size = torch.distributed.get_world_size()
                        p.grad.div_(float(world_size))
                    else:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
        return

    @torch.no_grad()
    def _grad_norm(self, by=None, weight_adaptive=False):
        # shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        if not by:
            norm = torch.norm(
                torch.stack([
                    ((torch.abs(p.data) if weight_adaptive else 1.0) * p.grad).norm(p=2)
                    for group in self.param_groups for p in group["params"]
                    if p.grad is not None
                ]),
                p=2
            )
        else:
            norm = torch.norm(
                torch.stack([
                    ((torch.abs(p.data) if weight_adaptive else 1.0) * self.state[p][by]).norm(p=2)
                    for group in self.param_groups for p in group["params"]
                    if p.grad is not None
                ]),
                p=2
            )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

    def maybe_no_sync(self):
        if torch.distributed.is_initialized():
            return self.model.no_sync()
        else:
            return contextlib.ExitStack()

    @torch.no_grad()
    def set_closure(self, loss_fn, inputs, targets, model, **kwargs):
        # create self.forward_backward_func, which is a function such that
        # self.forward_backward_func() automatically performs forward and backward passes.
        # This function does not take any arguments, and the inputs and targets data
        # should be pre-set in the definition of partial-function

        def get_grad():
            self.base_optimizer.zero_grad()
            with torch.enable_grad():
                # outputs = model(inputs)
                loss = loss_fn(inputs, targets, **kwargs)
                # loss = loss_fn(outputs, targets, **kwargs)
            loss_value = loss.data.clone().detach()
            loss.mean().backward() # max: modified to .mean() here
            return None, loss_value

        self.forward_backward_func = get_grad

    # @torch.no_grad()
    def step(self):
        inputs,targets,loss_fct,model,defined_backward,_ ,_ = self.paras
        # self.set_closure(loss_fct, inputs, targets, model)

        # get_grad = self.forward_backward_func
        #
        # with self.maybe_no_sync():
        #     # get gradient
        #     outputs, loss_value = get_grad()
        #     return_loss = loss_value.clone().detach()
        #     # perturb weights
        #     self.perturb_weights(rho=self.rho_t)
        #
        #     # disable running stats for second pass
        #     disable_running_stats(model)
        #
        #     # get gradient at perturbed weights
        #     get_grad()
        #
        #     # decompose and get new update direction
        #     self.gradient_decompose(self.alpha)
        #
        #     # unperturb
        #     self.unperturb()
        with model.no_sync():
            loss = loss_fct(inputs, targets)
            predictions = None
            return_loss = loss.clone().detach()
            self.returnthings = (predictions, return_loss)
            loss = loss.mean()
            defined_backward(loss)
            self.perturb_weights(rho=self.rho_t)
            disable_running_stats(model)

            loss = loss_fct(inputs, targets)
            loss = (loss).mean()
            defined_backward(loss)
            self.gradient_decompose(self.alpha)
            self.unperturb()

        # synchronize gradients across workers
        self._sync_grad()

        # update with new directions
        self.base_optimizer.step()

        # enable running stats
        enable_running_stats(model)

        self.returnthings = (None, return_loss)

        # return outputs, loss_value

def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)