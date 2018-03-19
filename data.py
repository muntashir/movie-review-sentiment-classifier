import os
import random
import json


class Dataset:

    def __init__(self, data_dir, force_rebuild=False):
        test_percent = 0.1
        validation_percent = 0.1

        # Index for minibatches
        self.data_index = {'train': 0, 'validation': 0}
        self.epoch_count = 0

        dataset_filepath = os.path.join(data_dir, 'dataset.json')
        if os.path.isfile(dataset_filepath) and not force_rebuild:
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

    def __load_text_as_vectors(self, batch):
        batch_size = len(batch)
        # TODO load text file as vector
        return batch

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
