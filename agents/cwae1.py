import torch
from torch.utils.data.dataloader import DataLoader
from utils.metric import AverageMeter
from metrics.acc import accumulate_acc
import models
from metrics import cw
from .default import NormalNN
import wandb
from copy import deepcopy


class CWAE1(NormalNN):

    def __init__(self, agent_config):
        super().__init__(agent_config)
        self.task_count = 0
        self.copy_prev_model = None

    def learn_batch(self, train_loader, val_loader=None):

        # 1.Learn the parameters for current task
        super().learn_batch(train_loader, val_loader)

        self.copy_prev_model = deepcopy(self.model)
        for p in self.copy_prev_model.parameters():
            p.requires_grad = False

        self.task_count += 1

    def criterion(self, inputs, targets, tasks, regularization=True, z_actv=None, prev_z_actv=None, **kwargs):
        loss = super().criterion(inputs, targets, tasks, **kwargs)
        mode = 'train' if self.training else 'valid'
        if self.wandb_is_on:
            wandb.log({f"{mode}/batch/task#{self.task_count}/loss": loss})

        if regularization and self.task_count >= 1:
            cw_loss = 0
            if self.config['mode'] == "CW" and prev_z_actv is not None:
                cw_loss += cw.cw_sampling_silverman(z_actv, prev_z_actv)
            # print(f"cw: {cw_loss}")
            loss += self.config['reg_coef'] * cw_loss
            if self.wandb_is_on:
                wandb.log({f"{mode}/batch/task#{self.task_count}/cw_loss": cw_loss})
        return loss

    def update_model(self, inputs, targets, tasks):
        z_actv, out = self.forward_once(inputs)
        prev_z_actv = None
        if self.copy_prev_model is not None:
            prev_z_actv = self.copy_prev_model.features(inputs)
        loss = self.criterion(out, targets, tasks, z_actv=z_actv, prev_z_actv=prev_z_actv)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach(), out

    def validation(self, dataloader):
        # This function doesn't distinguish tasks.
        acc = AverageMeter()
        losses = AverageMeter()
        task_name = ""
        for _, _, task in dataloader:
            task_name = task[0]
            break

        orig_mode = self.training
        self.eval()
        for i, (inputs, targets, task) in enumerate(dataloader):
            if self.gpu:
                with torch.no_grad():
                    inputs = inputs.cuda()
                    targets = targets.cuda()
            z_actv, output = self.predict_once(inputs)
            loss = self.criterion(output, targets, task, z_actv=z_actv)

            # Summarize the performance of all tasks, or 1 task, depends on dataloader.
            # Calculated by total number of data.
            acc = accumulate_acc(output, targets, task, acc)
            losses.update(loss, inputs.size(0))

        self.train(orig_mode)
        self.log(f"* VALID - Accuracy {acc.avg:.3f} Loss {losses.avg:.4f}")
        if self.wandb_is_on:
            wandb.log({f"valid/acc/task#{task_name}": acc.avg, f"valid/loss/task#{task_name}": losses.avg})
        return acc.avg

    def forward_once(self, x, prev_model=None):
        if prev_model is None:
            z_actv = self.model.features(x)
            out = self.model.logits(z_actv)
        else:
            z_actv = prev_model.features(x)
            out = prev_model.logits(z_actv)
        return z_actv, out

    def predict_once(self, inputs):
        self.model.eval()
        z_actv, out = self.forward_once(inputs)
        for t in out.keys():
            out[t] = out[t].detach()
        return z_actv, out
