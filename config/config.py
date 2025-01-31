import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# add save path?
class cfg:
    main_path = 'data/'
    train_dir = main_path + 'seg_train/seg_train/'
    test_dir = main_path + 'seg_test/seg_test/'
    classes = [
        'mountain', 'street', 'glacier',
        'buildings', 'sea', 'forest'
    ]
    onnx = True # save model as onnx
    
    model = "resnet50" # supports resnet18, 34 and 50 # do 101 also?
    image_size = 224
    batch_size = 64
    num_epochs = 30
    learning_rate = 1e-5
    weight_decay = 1e-4
    early_stopping = 5
    seed = 1
    
    out_channels = len(classes)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # transformers for trainig set
    p=0.25
    train_transform = A.Compose([
        A.Resize(width=image_size, height=image_size, p=1.0),
        A.HorizontalFlip(p=p),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=0, p=p),
        A.RandomGamma(gamma_limit = (40, 60), p=p),
        A.Rotate(limit=20, p=p),
        A.PixelDropout(dropout_prob=0.01, p=p),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    # transformer for validation set
    test_transform = A.Compose([
        A.Resize(width=image_size, height=image_size, p=1.0),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])