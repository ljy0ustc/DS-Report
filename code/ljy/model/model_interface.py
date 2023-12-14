import inspect
import torch
import importlib
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
from torch import nn

import pytorch_lightning as pl
from sklearn.metrics import f1_score


class MInterface(pl.LightningModule):
    def __init__(self, model_name, loss, lr, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()

    def forward(self, img):
        return self.model(img)

    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = self.loss_function(out.float(), batch['label'].float())
        self.log('lr', self.scheduler.optimizer.param_groups[0]['lr'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        self.valid_content={
            "label":[],
            "out":[]
        }

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        out = self(batch)
        loss = self.loss_function(out.float(), batch['label'].float())
        label_digit = batch['label'].argmax(axis=1)
        out_digit = out.argmax(axis=1)

        correct_num = sum(label_digit == out_digit).cpu().item()

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', correct_num/len(out_digit),
                 on_step=False, on_epoch=True, prog_bar=True)

        self.valid_content['label']+=list(label_digit.cpu().numpy())
        self.valid_content['out']+=list(out_digit.cpu().numpy())

    def on_validation_epoch_end(self):
        self.log('val_f1', f1_score(self.valid_content['label'],self.valid_content['out'],average="micro"),on_step=False, on_epoch=True, prog_bar=True)
        with open("data/ref/val_result.txt","w") as f:
            f.write("\n".join(['bot' if i else 'human' for i in self.valid_content['out']]))

    def on_test_epoch_start(self):
        self.test_content=[]
    
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        out = self(batch)
        out_digit = out.argmax(axis=1)
        self.test_content+=list(out_digit.cpu().numpy())

    def on_test_epoch_end(self):
        with open("data/ref/test_result.txt","w") as f:
            f.write("\n".join(['bot' if i else 'human' for i in self.test_content]))

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay)

        if self.hparams.lr_scheduler is None:
            return self.optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                self.scheduler = lrs.StepLR(self.optimizer,
                                       step_size=self.hparams.lr_decay_steps,
                                       gamma=self.hparams.lr_decay_rate)
            elif self.hparams.lr_scheduler == 'cosine':
                self.scheduler = lrs.CosineAnnealingLR(self.optimizer,
                                                  T_max=self.hparams.lr_decay_steps,
                                                  eta_min=self.hparams.lr_decay_min_lr)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [self.optimizer], [self.scheduler]

    def configure_loss(self):
        loss = self.hparams.loss.lower()
        if loss == 'bce':
            self.loss_function = nn.BCELoss(reduction='mean')
        else:
            raise ValueError("Invalid Loss Type!")
        
    def on_save_checkpoint(self, checkpoint):
        checkpoint.pop('optimizer_states')
        to_be_removed = []
        for key, value in checkpoint['state_dict'].items():
            try:
                if not self.get_parameter(key).requires_grad:
                    to_be_removed.append(key)
            except AttributeError:
                to_be_removed.append(key)
        for key in to_be_removed:
            checkpoint['state_dict'].pop(key)

    def load_model(self):
        name = self.hparams.model_name
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            Model = getattr(importlib.import_module(
                '.'+name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)