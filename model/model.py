
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from .sampler import GreedyCoresetSampler, ApproximateGreedyCoresetSampler, RandomSampler
from .neighborSerach import FaissNN, ApproximateFaissNN, NearestNeighbourScorer
from model.backbones import MyModel
from utils.utils import dotdict


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_bank = None
        self.label_bank = None
        self.class_indicator = None


    def load(self, args, device, setting):
        self.args = args
        self.device = device
        self.setting = setting
        self.backbone = MyModel(args, load=False).to(self.device)
        self.backbone_checkpoint_path = f"{args.backbone_path}/{setting}/checkpoint.pth"
        self.update_bank_counter = {args.update_bank_interv: 0}
        self._load_backbone_params()
        
        if args.nn_method == "faiss":
            self.nn_method = FaissNN(False, args.faiss_workers)
        else:
            self.nn_method = ApproximateFaissNN(False, args.faiss_workers)
        self.classifier = NearestNeighbourScorer(n_neighbours=args.n_neighbours, nn_method=self.nn_method)
        
        if args.sampler == "greedy_coreset":
            self.featuresampler = GreedyCoresetSampler(
                percentage = args.sample_percentage,
                device = self.device,
                num_classes = args.num_classes,
                dimension_to_project_features_to=args.d_features,
            )
        elif args.sampler == "approximate_coreset":
            self.featuresampler = ApproximateGreedyCoresetSampler(
                number_of_starting_points = args.n_starting_points,
                percentage = args.sample_percentage,
                device = self.device,
                num_classes = args.num_classes,
                dimension_to_project_features_to=args.d_features,
            )
        else:
            self.featuresampler = RandomSampler(percentage = args.sample_percentage)


    def _embed(self, batch_x, detach=True):
        
        self.backbone.eval()
        batch_x = batch_x.to(self.device).float()
        with torch.no_grad():
            features = self.backbone(batch_x)
        if detach:
            features = features.detach().cpu()
        return features
    
    def _embed_train(self, batch_x, batch_label, detach=True):
        
        self.backbone.eval()
        x_data = batch_x
        label_data = batch_label
        x_data = x_data.to(self.device).float()
        with torch.no_grad():
            features = self.backbone(x_data)
        if detach:
            features = features.detach().cpu()
        return features, label_data


    def fit(self, train_loader):
        self._fill_memory_bank_iter(train_loader)


    def _fill_memory_bank_iter(self, dataloader):

        total_features = []
        total_labels = []
        with tqdm.tqdm(
            dataloader, desc="Computing support features...", position=1, leave=False, ncols=50
        ) as data_iterator:
            for x, label in data_iterator:
                features, labels = self._embed_train(x, label)
                total_features.append(features)
                total_labels.append(labels)
                
        total_features = torch.cat(total_features, dim=0) # different stack size
        total_labels = torch.cat(total_labels, dim=0)
        self._update(total_features, total_labels)
        sampled_features, self.class_indicator = self.featuresampler.run(self.feature_bank, self.label_bank)

        self.classifier.fit(features=sampled_features)
        
        
    def _fill_memory_bank(self, x, label):

        features, labels = self._embed_train(x, label)
        self._update(features, labels)
        self.update_bank_counter[self.args.update_bank_interv] += 1
        if self.update_bank_counter[self.args.update_bank_interv] == self.args.update_bank_interv:
            self.update_bank_counter[self.args.update_bank_interv] = 0
            features, self.class_indicator = self.featuresampler.run(self.feature_bank, self.label_bank)
            self.classifier.fit(features=features)


    def _update(self, features, labels):
        if self.feature_bank is not None:
            self.feature_bank = torch.cat([self.feature_bank, features], dim=0)
        else:
            self.feature_bank = features
        if self.label_bank is not None:
            self.label_bank = torch.cat([self.label_bank, labels], dim=0)
        else:
            self.label_bank = labels
        
        
    def predict(self, test_loader):
        pred = []
        gt = []
        trained_feature_bank = self.feature_bank
        trained_label_bank = self.label_bank
        trained_class_indicator = self.class_indicator
        for x, label in test_loader:
            features = self._embed(x)
            pred.append(self.classifier.predict(features, self.class_indicator).cpu()) 
            if self.args.update_in_test:
                self._fill_memory_bank(x, label)
            gt.append(label)
        pred = torch.cat(pred, dim=0)  # [N]
        gt = torch.cat(gt, dim=0)  # [N]
        
        self.feature_bank = trained_feature_bank
        self.label_bank = trained_label_bank
        self.class_indicator = trained_class_indicator
        return pred, gt


    def _load_backbone_params(self):
        self.backbone.load_state_dict(torch.load(self.backbone_checkpoint_path))

    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "model_params.pkl")

    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_file_path = os.path.join(save_path, self.setting)
        if not os.path.exists(save_file_path):
            os.mkdir(save_file_path)
        self.classifier.save(
            save_file_path, save_features_separately=False, prepend=prepend
        )
        model_params = {
            "args_dict" : self.args.__dict__,
            "feature_bank": self.feature_bank,
            "label_bank": self.label_bank,
            "class_indicator": self.class_indicator,
        }
        with open(self._params_file(save_file_path, prepend), "wb") as save_file:
            pickle.dump(model_params, save_file, pickle.HIGHEST_PROTOCOL)

    def load_from_path(
        self,
        load_path: str,
        device: torch.device,
        setting: str,
        prepend: str = "",
    ) -> None:
        load_file_path = os.path.join(load_path, setting)
        with open(self._params_file(load_file_path, prepend), "rb") as load_file:
            model_params = pickle.load(load_file)
        self.load(
            dotdict(model_params['args_dict']), 
            device,
            setting
            )
        self.feature_bank = model_params['feature_bank']
        self.label_bank = model_params['label_bank']
        self.class_indicator = model_params['class_indicator']

        self.classifier.load(load_file_path, prepend)

