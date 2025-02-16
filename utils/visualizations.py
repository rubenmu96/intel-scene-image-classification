import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import albumentations as A
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from albumentations.pytorch.transforms import ToTensorV2
import torch

def pie_chart(train, test, classes, col_name="labels"):
    """
    This function creates a pie chart for training set and test set for
    showing the distribution of the classes

    Parameters:
    - train: DataFrame with a column "labels" containing the training set labels.
    - test: DataFrame with a column "labels" containing the test set labels.
    - classes: List of class names corresponding to the unique labels.
    - col_name: String with the name of the target variable
    """
    _, train_counts = np.unique(train[col_name], return_counts=True)
    _, test_counts = np.unique(test[col_name], return_counts=True)

    fig, axs = plt.subplots(1, 2, figsize=(12, 7))

    labels_with_train_counts = [f'{name} ({count})' for name, count in zip(classes, train_counts)]
    axs[0].pie(train_counts, explode=[0]*len(classes), labels=labels_with_train_counts, autopct='%1.2f%%')
    axs[0].set_title('Train set proportions', pad=20)

    labels_with_test_counts = [f'{name} ({count})' for name, count in zip(classes, test_counts)]
    axs[1].pie(test_counts, explode=[0]*len(classes), labels=labels_with_test_counts, autopct='%1.2f%%')
    axs[1].set_title('Validation set proportions', pad=20)

    plt.tight_layout()
    plt.show()


def display_nxn(images, labels, classes, nxn=5):
    '''
    Displays n x n images.

    Parameters:
    - images: dataframe with path to images
    - labels: dataframe with labels
    - classes: index/id to label name
    - nxn: (square) grid size of images
    '''
    figsize = np.max([10, nxn**2 // 2])
    
    if figsize > 15:
        figsize = 15

    plt.figure(figsize = (figsize, figsize))
    for i in range(nxn**2):
        image_label = classes[labels[i]]
        image = Image.open(images[i])
        image = image.convert('RGB')
        plt.subplot(nxn, nxn, i + 1)
        plt.imshow(image)
        plt.xticks([]), plt.yticks([])
        plt.title(image_label)
    plt.show()


def _add_transform(img, trans, cfg):
    """
    
    """
    base_transform = A.Compose([
        A.Resize(width=cfg.image_size, height=cfg.image_size, p=1.0),
        trans,
        A.Normalize(mean=cfg.mean, std=cfg.std),
        ToTensorV2(),
    ])

    augmented = base_transform(image=img)['image']
    transformed_image = augmented.permute(1, 2, 0).numpy()
    transformed_image = transformed_image * np.array(cfg.std) + np.array(cfg.mean)
    transformed_image = np.clip(transformed_image, 0, 1)
    return transformed_image

def visualize_transform(image, transforms, cfg):
    image = Image.open(image).convert('RGB')
    
    figsize = 4 * len(transforms)
    plt.figure(figsize=(figsize, figsize))
    plt.subplot(1, len(transforms) + 1 , 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    image = np.array(image)
    for idx, trans in enumerate(transforms):
        transformed_image = _add_transform(image, trans, cfg)

        plt.subplot(1, len(transforms) + 1, idx + 2)
        plt.imshow(transformed_image)
        plt.title(type(trans).__name__)
        plt.axis('off')
        
    plt.show()


def _get_all_predictions(model, loader, device, get_misclassified=False):
    model.eval()
    all_preds = []
    all_labels = []
    
    if get_misclassified: all_images = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if get_misclassified:
                all_images.append(images.cpu().numpy())

    if get_misclassified:
        return all_images, np.array(all_preds), np.array(all_labels)
    else:
        return np.array(all_labels), np.array(all_preds)

def plot_confusion_matrix(model, test_set, cfg):
    true_labels, pred_labels = _get_all_predictions(model, test_set, cfg.device)

    id2label = {i: class_name for i, class_name in enumerate(cfg.classes)}

    cm = confusion_matrix(true_labels, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=id2label.values())
    disp.plot(cmap=plt.cm.Blues, colorbar=False)


def classified_wrongly(model, test_set, cfg):
    images, labels, true = _get_all_predictions(
        model, test_set, cfg.device, get_misclassified=True
    )
    wrong = np.nonzero(labels != true)
    unbatch_images = {}
    i = 0
    for batches in images:
        for image in batches:
            unbatch_images[i] = image.transpose(1, 2, 0)
            i += 1
    
    return unbatch_images, wrong[0] , labels, true

def display_wrong_nn(images, labels, preds, classes, nxn=5):
    figsize = np.max([10, nxn**3 // 2])
    
    if figsize > 15:
        figsize = 15

    plt.figure(figsize = (figsize, figsize))
    for i in range(nxn**2):
        image_label = classes[labels[i]]
        pred_label = classes[preds[i]]
        image = images[i] * np.array(cfg.std) + np.array(cfg.mean)
        plt.subplot(nxn, nxn, i + 1)
        plt.imshow(image)
        plt.xticks([]), plt.yticks([])
        plt.title(f"True: {image_label}.\n Predicted: {pred_label}")
    plt.show()