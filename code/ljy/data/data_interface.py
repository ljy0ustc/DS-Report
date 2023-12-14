import inspect
import importlib
import pickle as pkl
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from PIL import Image
from transformers import AutoImageProcessor
import torch

class TrainCollater:
    def __init__(self,image_processor):
        self.image_processor = image_processor

    def __call__(self, batch):
        profile_image_path_list=[sample["profile_image"] for sample in batch]
        profile_banner_path_list=[sample["profile_banner"] for sample in batch]
        theme=torch.stack([torch.tensor(t['theme'],dtype=torch.float).view(1) for t in batch])
        label=torch.stack([torch.tensor(l['label'],dtype=torch.float) for l in batch])
        profile_image_list=[]
        profile_banner_list=[]
        for profile_image_path in profile_image_path_list:
            if len(profile_image_path)>0:
                try:
                    profile_image=Image.open(profile_image_path)
                    profile_image=profile_image.convert('RGB')
                except:
                    profile_image=Image.new('RGB', (224, 224))
            else:
                profile_image=Image.new('RGB', (224, 224))
            profile_image_list.append(profile_image)
        for profile_banner_path in profile_banner_path_list:
            if len(profile_banner_path)>0:
                try:
                    profile_banner=Image.open(profile_banner_path)
                    profile_banner=profile_banner.convert('RGB')
                except:
                    profile_banner=Image.new('RGB', (224, 224))
            else:
                profile_banner=Image.new('RGB', (224, 224))
            profile_banner_list.append(profile_banner)
        profile_image = self.image_processor(profile_image_list, return_tensors="pt")['pixel_values']
        profile_banner = self.image_processor(profile_banner_list, return_tensors="pt")['pixel_values']
        new_batch={
            'profile_image': profile_image,
            'profile_banner': profile_banner,
            'theme': theme,
            #'followers_count': torch.stack([torch.tensor(t['followers_count'],dtype=torch.float).view(1) for t in batch]),
            #'friends_count': torch.stack([torch.tensor(t['friends_count'],dtype=torch.float).view(1) for t in batch]),
            #'listed_count': torch.stack([torch.tensor(t['listed_count'],dtype=torch.float).view(1) for t in batch]),
            #'statuses_count': torch.stack([torch.tensor(t['statuses_count'],dtype=torch.float).view(1) for t in batch]),
            #'favourites_count': torch.stack([torch.tensor(t['favourites_count'],dtype=torch.float).view(1) for t in batch]),
            'num_features': torch.stack([torch.tensor(t['num_features'],dtype=torch.float) for t in batch]),
            'bool_features': torch.stack([torch.tensor(t['bool_features'],dtype=torch.float) for t in batch]),
            'label': label
        }
        return new_batch

class DInterface(pl.LightningDataModule):

    def __init__(self, num_workers=8,
                 dataset='comp_data',
                 **kwargs):
        super().__init__()
        self.num_workers = num_workers
        self.dataset = dataset
        self.kwargs = kwargs
        self.batch_size = kwargs['batch_size']
        self.image_processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.load_data_module()

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.trainset = self.instancialize(stage='train')
            self.valset = self.instancialize(stage='dev')

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.testset = self.instancialize(stage='test')

        # # If you need to balance your data using Pytorch Sampler,
        # # please uncomment the following lines.
    
        # with open('./data/ref/samples_weight.pkl', 'rb') as f:
        #     self.sample_weight = pkl.load(f)

    # def train_dataloader(self):
    #     sampler = WeightedRandomSampler(self.sample_weight, len(self.trainset)*20)
    #     return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, sampler = sampler)

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=TrainCollater(self.image_processor))

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, collate_fn=TrainCollater(self.image_processor))

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, collate_fn=TrainCollater(self.image_processor))

    def load_data_module(self):
        name = self.dataset
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            self.data_module = getattr(importlib.import_module(
                '.'+name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Dataset File Name or Invalid Class Name data.{name}.{camel_name}')

    def instancialize(self, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.kwargs.
        """
        class_args = inspect.getargspec(self.data_module.__init__).args[1:]
        inkeys = self.kwargs.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        args1.update(other_args)
        return self.data_module(**args1)
