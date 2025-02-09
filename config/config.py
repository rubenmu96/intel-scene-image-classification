import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# add save path?
# add weights? would make it easier to run more "dynamic"?
class cfg:
    main_path = 'data/'
    train_dir = main_path + 'seg_train/seg_train/'
    test_dir = main_path + 'seg_test/seg_test/'
    classes = [
        'mountain', 'street', 'glacier',
        'buildings', 'sea', 'forest'
    ]
    
    model = "resnet50" # supports resnet18-101
    onnx = True # save model as onnx
    checkpoint = None # load model from a checkpoint
    image_size = 224 # (w, h) = (224, 224)
    batch_size = 32 # number of images per batch
    epochs = 30 # number of epochs
    learning_rate = 1e-5 # learning rate
    weight_decay = 1e-4 # weight decay
    early_stopping = 5
    reset = True # reset early stopping counter if new best loss is found
    save_path = None # None will save as {model}_intelscene.pth
    seed = 1 # seed
    
    out_channels = len(classes) # number of classes
    # normalize using mean and std so that z = (x - mean) / std
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # transformers for trainig set
    prob = 0.25
    train_transform = A.Compose([
        A.Resize(width=image_size, height=image_size, p=1.0),
        A.HorizontalFlip(p=prob),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=0, p=prob),
        A.RandomGamma(gamma_limit = (40, 60), p=prob),
        A.Rotate(limit=20, p=prob),
        A.PixelDropout(dropout_prob=0.01, p=prob),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    # transformer for validation set
    test_transform = A.Compose([
        A.Resize(width=image_size, height=image_size, p=1.0),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])