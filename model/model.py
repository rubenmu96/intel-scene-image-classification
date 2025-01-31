import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models
import time

class ImageClassifier:
    def __init__(self, cfg, require_grad=True, multiple_gpus=True):
        self.modelname = cfg.model
        self.out = cfg.out_channels
        self.require_grad = require_grad
        self.multiple_gpus = multiple_gpus
        self.device = cfg.device
        
        self.model = self._initialize_model()
    
    def _initialize_model(self):
        model_func, weights = self._get_model_and_weights()
        model = model_func(weights=weights)

        for param in model.parameters():
            param.requires_grad = self.require_grad

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, self.out)
        
        if torch.cuda.device_count() > 1 and self.multiple_gpus:
            model = nn.DataParallel(model)
        
        return model.to(self.device)
    
    def _get_model_and_weights(self):
        model_mapping = {
            "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
            "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
            "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
        }
        return model_mapping[self.modelname]
    
    def get_model(self, model_path=False):
        if model_path:
            state_dict = torch.load(model_path, map_location=self.device)
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(state_dict)
        return self.model

    def save_best_model(self, name=None):
        if name is None:
            name = f"{self.modelname}_intelscene.pth"
        model_to_save = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        torch.save(model_to_save.state_dict(), name)
        
    def print_metrics(self, epoch_metrics):
        df = pd.DataFrame(epoch_metrics)
        if len(df) == 1:
            print(df.tail(1).to_markdown(index=False))
        else:
            print(df.tail(1).to_markdown(index=False).split('\n')[1])
            print(df.tail(1).to_markdown(index=False).split('\n')[2])

    def train(self, train_data, optimizer, criterion, tracker):
        total_loss = 0
        num_batches = 0

        self.model.train()
        tracker.reset()

        for image, label in train_data:
            image, label = image.to(self.device), label.to(self.device)

            optimizer.zero_grad()

            yhat = self.model(image)
            loss = criterion(yhat, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            tracker.update(yhat, label)

        return total_loss / num_batches, tracker.compute_metrics()

    def validate(self, valid_data, criterion, tracker):
        total_loss = 0
        num_batches = 0

        self.model.eval()
        tracker.reset()

        with torch.no_grad():
            for image, label in valid_data:
                image, label = image.to(self.device), label.to(self.device)

                yhat = self.model(image)
                loss = criterion(yhat, label)
            
                total_loss += loss.item()
                num_batches += 1
                tracker.update(yhat, label)

        return total_loss / num_batches, tracker.compute_metrics()

    def run(
        self,
        epochs,
        train_data,
        valid_data,
        criterion,
        optimizer,
        tracker,
        early_stopping=np.inf,
        printing=False
    ):
        best_loss = np.inf
        early_stopping_counter = 0
        epoch_metrics = []

        for n in range(epochs):
            start = time.time()
            train_loss, train_metrics = self.train(train_data, optimizer, criterion, tracker)
            end = time.time()
            valid_loss, valid_metrics = self.validate(valid_data, criterion, tracker)
            
            metrics = {
                'Epoch': n + 1,
                'Training time (m)': np.round((end - start) / 60, 2),
                'train_loss': train_loss,
                'train_accuracy': train_metrics["accuracy"],
                'train_f1': train_metrics['macro_f1'],
                'valid_loss': valid_loss,
                'valid_accuracy': valid_metrics["accuracy"],
                'valid_f1': valid_metrics['macro_f1']
            }

            epoch_metrics.append(metrics)
            
            if printing:
                self.print_metrics(epoch_metrics)

            if valid_loss < best_loss:
                best_loss = valid_loss
                self.save_best_model()
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= early_stopping:
                print("Early stopping triggered.")
                break

        return pd.DataFrame(epoch_metrics), self.model