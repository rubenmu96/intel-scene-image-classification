"""
How to run:
python train.py --model resnet50
"""
import argparse
import torch.nn as nn
import torch.optim as optim
from model import ImageClassifier
from config import cfg
from utils import (
    MetricCalculator,
    load_data,
    convert_onnx,
)

def update_cfg_from_args(args):
    for key, value in vars(args).items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)

def main(args):
    update_cfg_from_args(args)    
    train_loader, test_loader = load_data(cfg)

    classifier = ImageClassifier(
        cfg=cfg,
        out=len(cfg.classes),
        require_grad=True,
        multiple_gpus=True
    )

    tracker = MetricCalculator(len(cfg.classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        classifier.get_model().parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay
    )

    metric_data, trained_model = classifier.run(
        epochs=cfg.num_epochs,
        train_data=train_loader,
        valid_data=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        tracker=tracker,
        early_stopping=5,
        printing=True
    )
    # change path to log/training_log.csv or log/training_log_{model}.csv?
    metric_data.to_csv("training_log.csv", index=False)

    if cfg.onnx:
        convert_onnx(f"{cfg.model}.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # review parser arguments
    parser.add_argument('--model', type=str, default='resnet50', help='Model name to use')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--onnx', type=bool, default=True, help='If converting to onnx after training')
    # add parser for checkpoint?
    args = parser.parse_args()

    main(args)