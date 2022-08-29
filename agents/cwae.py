import torch
from torch.utils.data.dataloader import DataLoader
from utils.metric import AverageMeter
from metrics.acc import accumulate_acc
import models
from metrics import cw
from .default import NormalNN
import wandb


class CWAE(NormalNN):

    def __init__(self, agent_config):
        super().__init__(agent_config)
        self.task_count = 0
        self.cwae_online = self.config['cwae_online']
        self.generators = {}

    def create_generator(self):
        cfg = self.config
        generator = models.__dict__[cfg['generator_type']].__dict__[cfg['generator_name']](cfg['latent_size'])

        optimizer = torch.optim.Adam(generator.parameters(), lr=self.config['generator_lr'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[self.config['generator_epoch']], gamma=0.1)
        generator = generator.cuda()
        return generator, optimizer, scheduler

    def get_generator(self, task_id):
        if task_id in self.generators:
            return self.generators[task_id]
        self.generators[task_id] = self.create_generator()
        return self.generators[task_id]

    def train_encoder(self, train_loader, generator, optimizer, scheduler):
        self.log(f"{15 * '='} Train Encoder {15 * '='}")
        losses = AverageMeter()
        generator.train()
        for epoch in range(self.config['generator_epoch']):
            for y in train_loader:
                y = y.cuda()
                generator_loss = (cw.cw(generator(y)))
                optimizer.zero_grad()
                generator_loss.backward()
                optimizer.step()
                losses.update(generator_loss, y.size(0))

            self.log(f"Epoch: {epoch+1}  |  loss: {losses.avg:.4f}")
            scheduler.step()
        generator.eval()

    def train_generator(self, train_loader, generator, optimizer, scheduler):
        self.log(f"{15 * '='} Train Generator {15 * '='}")

        losses = AverageMeter()
        generator.train()
        for epoch in range(self.config['generator_epoch']):
            for y in train_loader:
                y = y.cuda()
                v = torch.randn((y.size(0), self.config['latent_size'])).cuda()

                generator_loss = cw.cw_sampling_silverman(y, generator(v))
                optimizer.zero_grad()
                generator_loss.backward()
                optimizer.step()

                losses.update(generator_loss, y.size(0))

            self.log(f"Epoch: {epoch+1}  |  loss: {losses.avg:.4f}")
            if self.wandb_is_on:
                wandb.log({f"generator/loss/task#{self.task_count}": losses.avg})
            scheduler.step()

        generator.eval()

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
        if self.wandb_is_on:
            wandb.log({"z_actv": wandb.Histogram(actv_data.cpu())})
        actv_loader = DataLoader(actv_data, batch_size=self.config['batch_size'], shuffle=True)

        # 3. train generator/encoder
        if self.cwae_online:
            generator, optimizer, scheduler = self.get_generator(self.task_count + 1)
        else:
            generator, optimizer, scheduler = self.get_generator(0)

        if self.config['mode'] == "GENERATOR":
            self.train_generator(actv_loader, generator, optimizer, scheduler)
        elif self.config['mode'] == "ENCODER":
            self.train_encoder(actv_loader, generator, optimizer, scheduler)
        else:
            raise ValueError("Incorrect mode")

        # 4. Unfroze network
        self.train(mode=mode)
        for name, param in self.model.named_parameters():
            param.requires_grad = True
            # if self.config['freeze_model']:
            #     param.requires_grad = False
            #     if 'last' in name or 'fc1' in name:
            #         param.requires_grad = True
            # else:
            #     param.requires_grad = True
        self.task_count += 1

    def criterion(self, inputs, targets, tasks, regularization=True, z_actv=None, **kwargs):
        loss = super().criterion(inputs, targets, tasks, **kwargs)
        mode = 'train' if self.training else 'valid'
        if self.wandb_is_on:
            wandb.log({f"{mode}/batch/task#{self.task_count}/loss": loss})
        if self.config['reg_coef_2']:
            norm = torch.norm(z_actv, p=2)
            loss += self.config['reg_coef_2'] * norm
            if self.wandb_is_on:
                wandb.log({f"{mode}/batch/task#{self.task_count}/norm": norm})

        if regularization and self.task_count >= 1:
            # print(f"z: {z_actv.size()}")
            v = torch.randn((z_actv.size(0), self.config['latent_size'])).cuda()
            # coeff = torch.as_tensor(self.config['reg_coef_2']).cuda() if self.config['reg_coef_2'] != 0 else None
            # cw_loss = torch.log(cw.cw_sampling_silverman(z_actv, self.generator(v)))
            cw_loss = 0
            for gen_id, (generator, _, _) in self.generators.items():
                if self.config['mode'] == "GENERATOR":
                    cw_loss += cw.cw_sampling_silverman(z_actv, generator(v)) / len(self.generators)
                elif self.config['mode'] == "ENCODER":
                    cw_loss += (cw.cw(generator(z_actv))) / len(self.generators)
                else:
                    raise ValueError("Incorrect mode")
            loss += self.config['reg_coef'] * cw_loss
            if self.wandb_is_on:
                wandb.log({f"{mode}/batch/task#{self.task_count}/cw_loss": cw_loss})
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

    def forward_once(self, x):
        z_actv = self.model.features(x)
        # print(f"zactv: {z_actv.size()}")
        out = self.model.logits(z_actv)
        return z_actv, out

    def predict_once(self, inputs):
        self.model.eval()
        z_actv, out = self.forward_once(inputs)
        for t in out.keys():
            out[t] = out[t].detach()
        return z_actv, out
