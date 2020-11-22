import os
import time
import itertools
import cv2
import numpy
from matplotlib import pyplot as plt

import torch

from tqdm import tqdm
from tensorboardX import SummaryWriter

class ValueAverager:
    """For accumulating and then calculating the average value"""
    def __init__ (self):
        self.accum_value = 0
        self.num_values = 0

    def add(self, value):
        self.accum_value += value
        self.num_values += 1

    @property
    def average(self):
        return self.accum_value / self.num_values if self.num_values else 0


class SummaryWriterForTesting:
    """This is just for testing a simple writer so I can check if anything
       broke when refactoring code.  Assumes one simplified the NN analysis 
       enough to make it useful"""
    def __init__ (self):
        import collections
        self.scalars = collections.defaultdict(list)
        self.texts = dict()

        self.logdir="C:/temp"

    def add_scalar(self, value_name, value, step):
        self.scalars[value_name] += (float(value), step)

    def add_text(self, tag, text):
        self.texts[tag] = text

    def add_figure (self, *args, **kwds):
        pass

    def add_image (self, *args, **kwds):
        pass

    def close(self):
        import json
        with open("new.txt", 'w') as fp:
            json.dump(self.scalars, fp, indent=4, sort_keys=True)
            json.dump(self.texts, fp, indent=4, sort_keys=True)

class ImageClassificatioanReport:
    max_num_incorrect_images = 56

    class EpochStats:
        def __init__(self):
            self.loss = ValueAverager()
            self.accuracy = ValueAverager()

    def __init__ (self, model, label_names, report_every="epoch", save_best_epochs=True, enabled=True):
        self.writer = SummaryWriter() if enabled else None
        self.model = model
        self.label_names = label_names
        self.report_every = report_every
        self._is_report_step = False
        self.save_best_epochs = save_best_epochs

    @property
    def is_report_step(self):
        return self._is_report_step
        
    def track_epochs (self, epoch_range):
        run_start = time.time()
        self.training_steps = 0
        self._is_report_step = False
        self.best_test_accuracy = 0
        most_accurate_file = None

        for epoch in epoch_range:
            self.epoch = epoch
            self.epoch_stats = self.EpochStats()

            start_time = time.time()
            yield self.epoch
            duration = time.time() - start_time

            if self.writer:
                self.writer.add_scalar ("Epoch Durations", duration, epoch)

            if self.save_best_epochs and self.test_stats.accuracy.average > self.best_test_accuracy:
                self.best_test_accuracy = self.test_stats.accuracy.average
                most_accurate_file = os.path.abspath(os.path.join(self.writer.logdir, f"epoch_{epoch}.pt"))
                self.save_model(most_accurate_file, epoch=epoch, accuracy=self.best_test_accuracy)

            print(f"Epoch {epoch}: Complete (trainAcc = {self.epoch_stats.accuracy.average:.2f}%,",
                                          f"trainLoss = {self.epoch_stats.loss.average:.3f},",
                                          f"testAcc = {self.test_stats.accuracy.average:.2f}%,",
                                          f"testLoss = {self.test_stats.loss.average:.3f},",
                                          f"duration = {duration:.2f}s)")

        print ("Time Taken:", time.time() - run_start)
        if most_accurate_file:
            print ("Most Accurate Model:", most_accurate_file)

    def track_train_batches (self, batches):
        for batch_step, batch in enumerate(tqdm(batches, desc = f"Epoch {self.epoch}", leave=False), start=1):
            self.training_steps += 1
            if self.writer and self.report_every != "epoch":
                self._is_report_step = self.training_steps % self.report_every == 0 or batch_step == len(batches)
            else:
                self._is_report_step = batch_step == len(batches)
            yield batch

    def add_batch_train_loss(self, loss):
        self.epoch_stats.loss.add(loss)
        if self._is_report_step and self.writer:
            self.writer.add_scalar("Loss/train", float(loss), self.training_steps)

    def add_batch_train_accuracy(self, accuracy):
        self.epoch_stats.accuracy.add(accuracy*100)
        if self._is_report_step and self.writer:
            self.writer.add_scalar("Acc/train", accuracy*100, self.training_steps)

    def track_test_batches (self, batches):
        self.test_stats = self.EpochStats()
        self.test_predictions = None
        self.test_labels = None
        self.incorrect_test_images = None

        for batch in batches:
            yield batch

        if self.writer:
            self.writer.add_scalar("Loss/test", self.test_stats.loss.average, self.training_steps)
            self.writer.add_scalar("Acc/test", self.test_stats.accuracy.average, self.training_steps)

            if self.test_predictions is not None:
                self._add_confusion_matrix()

            if self.incorrect_test_images is not None:
                self._add_incorrect_test_images()

    def add_batch_test_loss(self, loss):
        self.test_stats.loss.add(loss)

    def add_batch_test_accuracy(self, accuracy):
        self.test_stats.accuracy.add(accuracy*100)

    # Maybe create a class for batch test data and for train test data?  To simplify complexity
    # of language and it'll simplify classes, too..
    def add_batch_test_confusion_data(self, predictions, labels):
        if self.test_predictions is None:
            self.test_predictions = predictions.cpu().numpy()
            self.test_labels = labels.cpu().numpy()
        else:
            self.test_predictions = numpy.append(self.test_predictions, predictions.cpu().numpy())
            self.test_labels = numpy.append(self.test_labels, labels.cpu().numpy())

    def _add_confusion_matrix(self):
        import sklearn.metrics
        cm = sklearn.metrics.confusion_matrix(self.test_labels, self.test_predictions)
        figure = self._plot_confusion_matrix(cm)
        self.writer.add_figure("Confusion Matrix", figure, self.training_steps)

    def _plot_confusion_matrix(self, cm):
        """Returns a matplotlib figure containing the plotted confusion matrix.
        """
        figure = plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = numpy.arange(len(self.label_names))
        plt.xticks(tick_marks, self.label_names, rotation=45)
        plt.yticks(tick_marks, self.label_names)

        # Compute the labels from the normalized confusion matrix.
        labels = numpy.around(cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis], decimals=2)

        # Use white text if squares are dark; otherwise black.
        threshold = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        return figure
                
    def add_text (self, tag, text):
        if not self.writer: return
        text = self._convert_text_to_markdown(str(text))
        self.writer.add_text(tag, text)

    def _convert_text_to_markdown(self, text):
        return text.replace("\n", "  \n")

    def add_incorrect_test_images(self, correct_flags, images):
        if (self.incorrect_test_images is not None and 
            len(self.incorrect_test_images) > self.max_num_incorrect_images): 
            return

        incorrect_flags = correct_flags.logical_not()
        incorrect_images = images[incorrect_flags]

        if self.incorrect_test_images is None:
            self.incorrect_test_images = incorrect_images
        else:
            self.incorrect_test_images = torch.cat((self.incorrect_test_images, incorrect_images))

    def _add_incorrect_test_images(self):
        import torchvision.utils
        grid = torchvision.utils.make_grid(self.incorrect_test_images[:self.max_num_incorrect_images])
        self.writer.add_image(f"Incorrect Images (first {self.max_num_incorrect_images})", 
            grid, self.training_steps)

    def save_model (self, filename, **kwds):
        data_to_save = dict(
            state_dict = self.model.state_dict(),
            **kwds
        )
        filename = os.path.join(self.writer.logdir, filename)
        torch.save(data_to_save, filename)
