from .models import CNNClassifier, save_model, load_model
from .utils import ConfusionMatrix, accuracy, load_data, LABEL_NAMES
import torch
import torchvision
import torch.utils.tensorboard as tb
from torchvision.transforms import v2


def train(args):
    from os import path
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))

    """
    Your code here, modify your HW1 code

    """
    import torch

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = CNNClassifier().to(device)
    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th')))

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    loss = torch.nn.CrossEntropyLoss()

    train_data = load_data('data/train')
    valid_data = load_data('data/valid')

    global_step = 0
    for epoch in range(args.num_epoch):
        model.train()
        acc_vals = []
        for img, label in train_data:
            img, label = img.to(device), label.to(device)
            model(img)
            transforms  = v2.Compose([
                v2.RandomHorizontalFlip(p=1 ),
                v2.ColorJitter(brightness=(0.5,1.5),contrast=(1),saturation=(0.5,1.5),hue=(-0.1,0.1)),
                v2.ToDtype(torch.float32, scale=True),
            ])
            img = transforms(img)
            logit = model(img)
            loss_val = loss(logit, label)
            acc_val = accuracy(logit, label)

            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)
            acc_vals.append(acc_val.detach().cpu().numpy())

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1
        avg_acc = sum(acc_vals) / len(acc_vals)

        if train_logger:
            train_logger.add_scalar('accuracy', avg_acc, global_step)

        model.eval()
        acc_vals = []
        for img, label in valid_data:
            img, label = img.to(device), label.to(device)
            acc_vals.append(accuracy(model(img), label).detach().cpu().numpy())
        avg_vacc = sum(acc_vals) / len(acc_vals)

        if valid_logger:
            valid_logger.add_scalar('accuracy', avg_vacc, global_step)

        if valid_logger is None or train_logger is None:
            print('epoch %-3d \t acc = %0.3f \t val acc = %0.3f' % (epoch, avg_acc, avg_vacc))
        save_model(model)
    save_model(model)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-n', '--num_epoch', type=int, default=50)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.94e-3)
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
