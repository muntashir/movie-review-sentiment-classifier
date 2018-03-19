import model
import data
import sys
import argparse
import os

import torch
import torch.nn as nn
from torch.autograd import Variable


def main(args):
    dataset = data.Dataset(args.data_dir)
    criterion = nn.CrossEntropyLoss()

    net = model.Net(data.VOCAB_SIZE)

    if os.path.isfile(args.model_filename):
        checkpoint = (torch.load(filename) if (torch.cuda.is_available())
                      else torch.load(filename, map_location=lambda storage, loc: storage))
        net.load_state_dict(checkpoint['state_dict'])

    if (torch.cuda.is_available()):
        net.cuda()

    print('\nRunning test')
    epoch_end = False
    total_loss = []

    test_step = 0
    test_size = len(dataset.dataset['test'])
    while epoch_end == False:
        test_step += 1
        minibatch, epoch_end = dataset.get_next_minibatch('test', 1)
        batch_tensor = Variable(minibatch[0])
        labels_tensor = Variable(minibatch[1])

        if (torch.cuda.is_available()):
            batch_tensor = batch_tensor.cuda()
            labels_tensor = labels_tensor.cuda()

        output = net.forward(batch_tensor)
        loss = criterion(output, torch.max(labels_tensor, 1)[1])
        total_loss.append(loss.data)
        sys.stdout.write('Test step %i/%i\r' % (test_step, test_size))
        sys.stdout.flush()

    total_loss = float(sum(total_loss)[0]) / float(len(total_loss))
    print('\nTest loss: %f' % total_loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Runs the model on the test split')
    parser.add_argument(
        '-d',
        '--test-data',
        help='Directory where the test data is loaded from',
        default='dataset/',
        dest='data_dir')
    parser.add_argument(
        '-m',
        '--model-file',
        help='Filename of the saved model',
        default='checkpoint.model',
        dest='model_filename')
    main(parser.parse_args())
