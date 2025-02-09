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
        require_grad=True,
        multiple_gpus=True
    )
    if args.checkpoint is not None:
        classifier.get_model(args.checkpoint)

    tracker = MetricCalculator(len(cfg.classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        classifier.model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay
    )

    metric_data, trained_model = classifier.run(
        train_data=train_loader,
        valid_data=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        tracker=tracker,
        printing=True
    )
    metric_data.to_csv("log/training_log.csv", index=False)

    if cfg.onnx:
        convert_onnx(f"{cfg.model}.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=cfg.model, help='Model name to use')
    parser.add_argument('--batch_size', type=int, default=cfg.batch_size, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=cfg.learning_rate, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=cfg.num_epochs, help='Epochs')
    parser.add_argument('--onnx', type=bool, default=cfg.onnx, help='Convert best model to onnx')
    parser.add_argument('--checkpoint', type=str, default=None, help='Train from checkpoint')
    args = parser.parse_args()

    main(args)