import torch
from torch.utils.data.dataloader import DataLoader
from utils.metric import AverageMeter
from metrics.acc import accumulate_acc
import models
from metrics import cw
from .default import NormalNN


class CWAE(NormalNN):

    def __init__(self, agent_config):
        super().__init__(agent_config)
        self.task_count = 0
        self.generator = self.create_generator()

    def create_generator(self):
        cfg = self.config
        generator = models.__dict__[cfg['generator_type']].__dict__[cfg['generator_name']](cfg['latent_size'])
        return generator

    def train_generator(self, train_loader):
        self.log(f"{15 * '='} Train Generator {15 * '='}")
        generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.config['generator_lr'])
        self.generator = self.generator.cuda()
        losses = AverageMeter()

        for epoch in range(self.config['generator_epoch']):
            for y in train_loader:
                y = y.cuda()
                v = torch.randn((y.size(0), self.config['latent_size'])).cuda()

                generator_loss = cw.cw_sampling(y, self.generator(v))
                generator_optimizer.zero_grad()
                generator_loss.backward()
                generator_optimizer.step()

                losses.update(generator_loss, y.size(0))

            self.log(f"Epoch: {epoch}  |  loss: {losses.avg:.4f}")

    def learn_batch(self, train_loader, val_loader=None):

        # 1.Learn the parameters for current task
        super().learn_batch(train_loader, val_loader)

        # 2. Froze network and create activation data
        mode = self.training
        self.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        actv_data = torch.cat([self.model.features(inputs.cuda()) for inputs, _, _ in train_loader], dim=0)
        self.log(f"mean: {actv_data.mean()} std: {actv_data.std()}")
        actv_loader = DataLoader(actv_data, batch_size=self.config['batch_size'], shuffle=True)

        # 3. train generator
        self.train_generator(actv_loader)

        # 4. Unfroze network
        self.train(mode=mode)
        for p in self.model.parameters():
            p.requires_grad = True
        self.task_count += 1

    def criterion(self, inputs, targets, tasks, regularization=True, z_actv=None, **kwargs):
        loss = super().criterion(inputs, targets, tasks, **kwargs)
        if self.task_count >= 1:
            v = torch.randn((z_actv.size(0), self.config['latent_size'])).cuda()
            loss += self.config['reg_coef'] * cw.cw_sampling(z_actv, self.generator(v))
        return loss

    def update_model(self, inputs, targets, tasks):
        z_actv, out = self.forward_once(inputs)
        loss = self.criterion(out, targets, tasks, z_actv=z_actv)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach(), out

    def validation(self, dataloader):
        # This function doesn't distinguish tasks.
        acc = AverageMeter()
        losses = AverageMeter()

        orig_mode = self.training
        self.eval()
        for i, (inputs, targets, task) in enumerate(dataloader):
            if self.gpu:
                with torch.no_grad():
                    inputs = inputs.cuda()
                    targets = targets.cuda()
            z_actv, output = self.predict_once(inputs)
            if self.task_count >= 1:
                loss = self.criterion(output, targets, task, z_actv=z_actv)
            else:
                loss = self.criterion(output, targets, task)

            # Summarize the performance of all tasks, or 1 task, depends on dataloader.
            # Calculated by total number of data.
            acc = accumulate_acc(output, targets, task, acc)
            losses.update(loss, inputs.size(0))

        self.train(orig_mode)
        self.log(f"* VALID - Accuracy {acc.avg:.3f} Loss {losses.avg:.4f}")
        return acc.avg

    def forward_once(self, x):
        z_actv = self.model.features(x)
        out = self.model.logits(z_actv)
        return z_actv, out

    def predict_once(self, inputs):
        self.model.eval()
        z_actv, out = self.forward_once(inputs)
        for t in out.keys():
            out[t] = out[t].detach()
        return z_actv, out
