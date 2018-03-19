import model
import data
import sys
import argparse
import os
import itertools

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

def main(args):
    dataset = data.Dataset(args.data_dir)
    criterion = nn.CrossEntropyLoss()

    net = model.Net(data.VOCAB_SIZE)
    optimizer = optim.SparseAdam(net.parameters())
    step = 0

    if os.path.isfile(args.model_filename):
        checkpoint = torch.load(args.model_filename)
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optim'])
        step = checkpoint['step']
    else:
        if os.path.isfile(args.log_file):
            os.remove(args.log_file)

    if (torch.cuda.is_available()):
        net.cuda()

    for i in itertools.count():
        if ((step + i) % args.checkpoint_interval == 0):
            epoch_end = False
            total_loss = []

            while epoch_end == False:
                minibatch, epoch_end = dataset.get_next_minibatch('validation', 1)
                batch_tensor = Variable(minibatch[0])
                labels_tensor = Variable(minibatch[1])

                if (torch.cuda.is_available()):
                    batch_tensor = batch_tensor.cuda()
                    labels_tensor = labels_tensor.cuda()

                output = net.forward(batch_tensor)
                loss = criterion(output, torch.max(labels_tensor, 1)[1])
                total_loss.append(loss.data)

            total_loss = float(sum(total_loss)[0]) / float(len(total_loss))
            print('\nValidation loss: %f' % total_loss)

            with open(args.log_file, 'a') as log_file:
                log_file.write(str(step + i) +  "," + str(total_loss) + "\n")

            will_save = True
            if os.path.isfile(args.model_filename):
                checkpoint = torch.load(args.model_filename)
                old_loss = checkpoint['loss']
                if old_loss < total_loss:
                    will_save = False

            if will_save:
                torch.save({
                    'step': step + i,
                    'state_dict': net.state_dict(),
                    'loss': total_loss,
                    'optim': optimizer.state_dict()
                }, args.model_filename)

        minibatch, epoch_end = dataset.get_next_minibatch(
            'train', args.batch_size)

        batch_tensor = Variable(minibatch[0])
        labels_tensor = Variable(minibatch[1])

        sys.stdout.write('Training step %i\r' % (step+ i))
        sys.stdout.flush()

        if (torch.cuda.is_available()):
            batch_tensor = batch_tensor.cuda()
            labels_tensor = labels_tensor.cuda()

        optimizer.zero_grad()
        output = net.forward(batch_tensor)
        loss = criterion(output, torch.max(labels_tensor, 1)[1])
        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains the model')
    parser.add_argument(
        '-d',
        '--training-data',
        help='Directory where the training data is loaded from',
        default='dataset/',
        dest='data_dir')
    parser.add_argument(
        '-c',
        '--checkpoint-interval',
        help='Checkpoint training after these many steps',
        default=100,
        type=int,
        dest='checkpoint_interval')
    parser.add_argument(
        '-b',
        '--batch-size',
        help='The batch size',
        default=3,
        type=int,
        dest='batch_size')
    parser.add_argument(
        '-m',
        '--model-file',
        help='Filename of the saved model',
        default='checkpoint.model',
        dest='model_filename')
    parser.add_argument(
        '-l',
        '--log-file',
        help='Filename of the log',
        default='log.csv',
        dest='log_file')
    main(parser.parse_args())
