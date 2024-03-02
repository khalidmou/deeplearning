import torch
import numpy as np
from datetime import datetime
import torch.nn as nn
from .models import FCN, save_model
import torch.optim.lr_scheduler as lr_scheduler 
from .utils import load_dense_data,accuracy ,DENSE_CLASS_DISTRIBUTION, ConfusionMatrix
from . import dense_transforms
import torch.utils.tensorboard as tb
from torchvision.transforms import v2


def train(args):
    from os import path
    criterion = nn.CrossEntropyLoss()
    model = FCN()
    global_step = 0
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.000095)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=30)
    num_epoch = 40
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = load_dense_data('dense_data/train')
    valid_data = load_dense_data('dense_data/valid')
    train_logger, valid_logger = None, None
    if args.log_dir is not None:

        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)
    for epoch in range(num_epoch):
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)
        acc_vals = []
        validation = []
        model.train()
        for img, label in train_data:
            img, label = img.to(device), label.to(device)
            logit = model(img)
            loss_val = criterion(logit, label.long())
            acc_val = accuracy(logit, label)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1
            acc_vals.append(acc_val.detach().cpu().numpy())
        scheduler.step()
        avg_acc = sum(acc_vals) / len(acc_vals)
        if valid_logger:
            valid_logger.add_scalar('accuracy', avg_acc, global_step)
        model.eval()
        for img, label in valid_data:
            img, label = img.to(device), label.to(device)
            validation.append(accuracy(model(img), label).detach().cpu().numpy())
        validation_acc = sum(validation) / len(validation)
        if valid_logger: 
            valid_logger.add_scalar('accuracy', avg_vacc, global_step)

        if valid_logger:
            valid_logger.add_scalar('accuracy', avg_vacc, global_step)

        if valid_logger is None or train_logger is None:
            print('epoch %-3d \t acc = %0.3f \t val acc = %0.3f ' % (epoch, avg_acc, validation_acc))
        save_model(model)
    save_model(model)


    """
    Your code here, modify your HW1 / HW2 code
    Hint: Use ConfusionMatrix, ConfusionMatrix.add(logit.argmax(1), label), ConfusionMatrix.iou to compute
          the overall IoU, where label are the batch labels, and logit are the logits of your classifier.
    Hint: If you found a good data augmentation parameters for the CNN, use them here too. Use dense_transforms
    Hint: Use the log function below to debug and visualize your model
    """
    save_model(model)


def log(logger, imgs, lbls, logits, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    lbls: semantic label tensor
    logits: predicted logits tensor
    global_step: iteration
    """
    logger.add_image('image', imgs[0], global_step)
    logger.add_image('label', np.array(dense_transforms.label_to_pil_image(lbls[0].cpu()).
                                             convert('RGB')), global_step, dataformats='HWC')
    logger.add_image('prediction', np.array(dense_transforms.
                                                  label_to_pil_image(logits[0].argmax(dim=0).cpu()).
                                                  convert('RGB')), global_step, dataformats='HWC')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
