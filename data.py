import os
import random
import json
import torch
import numpy as np

VOCAB_SIZE = 89528
MAX_LENGTH = 100

class Dataset:

    def __init__(self, data_dir):
        test_percent = 0.05
        validation_percent = 0.05

        # Index for minibatches
        self.data_index = {'train': 0, 'validation': 0}
        self.epoch_count = 0

        dataset_filepath = os.path.join(data_dir, 'dataset.json')
        if os.path.isfile(dataset_filepath):
            print('Loading data split from cache')

            with open(dataset_filepath) as dataset_file:
                self.dataset = json.load(dataset_file)
        else:
            print('Generating data split')
            data_and_labels = []

            for folder, _, filenames in os.walk(data_dir):
                for filename in filenames:
                    if data_dir == folder:
                        continue

                    label = folder.split(os.sep)[-1]
                    full_path = os.path.join(folder, filename)
                    data_and_label = {}
                    data_and_label['path'] = full_path
                    data_and_label['label'] = label
                    data_and_labels.append(data_and_label)

            random.shuffle(data_and_labels)

            test_slice = int(len(data_and_labels) * test_percent)
            validation_slice = -int(len(data_and_labels) * validation_percent)

            self.dataset = {}
            self.dataset['test'] = data_and_labels[:test_slice]
            self.dataset['train'] = data_and_labels[test_slice:validation_slice]
            self.dataset['validation'] = data_and_labels[validation_slice:]

            with open(dataset_filepath, 'w') as dataset_file:
                json.dump(self.dataset, dataset_file)

        vocab_filepath = os.path.join(data_dir, 'imdb.vocab')
        if not os.path.isfile(vocab_filepath):
            print('vocab.txt file missing in dataset/')
        else:
            with open(vocab_filepath, 'r') as vocab_file:
                self.word_to_index = [line.rstrip('\n') for line in vocab_file]

    def __load_text_as_vectors(self, batch):
        batch_size = len(batch)
        vectors_and_labels = []
        time_steps = 0

        for data in batch:
            vectors_and_label = {}
            label = np.zeros(2)
            if (data['label'] == 'pos'):
                label[0] = 1
            elif (data['label'] == 'neg'):
                label[1] = 1
            vectors_and_label['label'] = label
            vectors_and_label['vectors'] = []

            filepath = data['path']
            with open(filepath, 'r') as f:
                words = f.read() \
                    .replace('<br />', ' ') \
                    .replace('(', '') \
                    .replace(')', '') \
                    .replace('--', '') \
                    .replace('.', ' ') \
                    .replace('"', ' ') \
                    .replace('\'', ' ') \
                    .replace('!', '') \
                    .replace('?', '') \
                    .replace('_', '') \
                    .replace('/', '') \
                    .replace(',', '') \
                    .replace(':', '') \
                    .replace(';', '') \
                    .replace('*', '') \
                    .replace('`', '') \
                    .replace('&', '') \
                    .replace('\\', '') \
                    .split(' ')
            words = list(filter(None, words))
            words = list(filter(lambda x: x != '-', words))

            for word in words:
                word = word.lower()
                try:
                    index = self.word_to_index.index(word)
                except ValueError:
                    if __name__ == '__main__':
                        print('Unknown word: ' + word)
                    index = self.word_to_index.index('UNKNOWN_WORD_TOKEN')
                word_vector = np.zeros(VOCAB_SIZE)
                word_vector[index] = 1
                vectors_and_label['vectors'].append(word_vector)

            time_steps = np.max([len(vectors_and_label['vectors']), time_steps])
            vectors_and_labels.append(vectors_and_label)

        batch_matrix = torch.zeros(batch_size, int(np.min([time_steps, MAX_LENGTH])), VOCAB_SIZE)
        label_matrix = torch.zeros(batch_size, 2).type(torch.LongTensor)

        for batch_number, vectors_and_label in enumerate(vectors_and_labels):
            vectors = vectors_and_label['vectors']
            # Pad vectors to max length in batch and limit to MAX_LENGTH
            vectors += [np.zeros(VOCAB_SIZE)] * (time_steps - len(vectors))
            if time_steps > MAX_LENGTH:
                vectors = vectors[:MAX_LENGTH]

            label = vectors_and_label['label']
            label_matrix[batch_number, :] = torch.from_numpy(label).type(torch.LongTensor)

            for time_step, vector in enumerate(vectors):
                batch_matrix[batch_number, time_step, :] = torch.from_numpy(vector)

        return batch_matrix, label_matrix

    def get_next_minibatch(self, dataset_split, batch_size):
        epoch_end = False

        if self.data_index[dataset_split] == 0:
            random.shuffle(self.dataset[dataset_split])
            if dataset_split == 'train':
                self.epoch_count += 1
                print('\nEpoch %i' % self.epoch_count)

        start_pos = self.data_index[dataset_split]
        end_pos = start_pos + batch_size

        if end_pos >= len(self.dataset[dataset_split]):
            end_pos = len(self.dataset[dataset_split])
            self.data_index[dataset_split] = 0
            epoch_end = True
        else:
            self.data_index[dataset_split] += batch_size

        minibatch = self.dataset[dataset_split][start_pos:end_pos]
        return self.__load_text_as_vectors(minibatch), epoch_end


def test():
    dataset = Dataset('dataset')
    assert(len(dataset.word_to_index) == VOCAB_SIZE)

    minibatch = dataset.get_next_minibatch('train', 3)
    assert(minibatch[0][0].size()[0] == 3)
    assert(minibatch[0][0].size()[2] == VOCAB_SIZE)
    assert(minibatch[0][1].size()[0] == 3)
    assert(minibatch[0][1].size()[1] == 2)

if __name__ == '__main__':
    test()
