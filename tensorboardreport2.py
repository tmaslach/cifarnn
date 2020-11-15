import time
import itertools
import cv2
import numpy
from matplotlib import pyplot as plt

import torch

from tqdm import tqdm
from tensorboardX import SummaryWriter

class TenserBoardReport:
    max_num_incorrect_images = 56

    class EpochStats:
        loss = 0
        accuracy = 0

    def __init__ (self, label_names, disable=False):
        self.writer = SummaryWriter() if not disable else None
        self.label_names = label_names
        self.incorrect_test_images = None

    def printable_step (self, print_every):
        """Returns true if training step is divisible by print_every or last batch
           step of epoch"""
        return self.training_steps % print_every == 0 or self.last_train_batch

    def track_epochs (self, epoch_range):
        run_start = time.time()
        self.training_steps = 0

        for epoch in epoch_range:
            self.epoch = epoch
            self.epoch_stats = self.EpochStats()

            start_time = time.time()
            start_step = self.training_steps
            yield self.epoch
            duration = time.time() - start_time
            steps = self.training_steps - start_step

            if self.writer:
                self.writer.add_scalar ("Epoch Durations", duration, epoch)
                epoch_loss = self.epoch_stats.loss / steps
                epoch_accuracy = self.epoch_stats.accuracy / steps * 100
                
                print(f"Epoch {epoch}: Complete (trainAcc = {epoch_accuracy:.2f}%,",
                                            f"trainLoss = {epoch_loss:.3f},",
                                            f"testAccuracy = {self.last_test_accuracy:.2f}%,",
                                            f"testLoss = {self.last_test_loss:.3f},",
                                            f"duration = {duration:.2f}s)")

        print ("Time Taken:", time.time() - run_start)

    def track_train_batches (self, batches):
        self.cur_train_loss = None
        self.cur_train_accuracy = None

        for batch_step, batch in enumerate(tqdm(batches, desc = f"Epoch {self.epoch}", leave=False), start=1):
            self.training_steps += 1
            self.last_train_batch = batch_step == len(batches)
            yield batch

            if self.writer:
                if self.cur_train_loss is not None:
                    self.writer.add_scalar("Loss/train", float(self.cur_train_loss), self.training_steps)
                if self.cur_train_accuracy is not None:
                    self.writer.add_scalar("Acc/train", self.cur_train_accuracy*100, self.training_steps)

    def set_batch_train_loss(self, loss):
        self.cur_train_loss = loss
        self.epoch_stats.loss += loss

    def set_batch_train_accuracy(self, accuracy):
        self.cur_train_accuracy = accuracy
        self.epoch_stats.accuracy += accuracy

    def track_test_batches (self, batches):
        self.batch_test_loss = 0
        self.batch_test_accuracy = 0
        self.batch_test_predictions = None
        self.batch_test_labels = None
        self.incorrect_test_images = None

        for batch in batches:
            yield batch

        self.last_test_loss = self.batch_test_loss / len(batches)
        self.last_test_accuracy = self.batch_test_accuracy / len(batches) * 100

        if self.writer:
            self.writer.add_scalar("Loss/test", self.last_test_loss, self.training_steps)
            self.writer.add_scalar("Acc/test", self.last_test_accuracy, self.training_steps)

            if self.batch_test_predictions is not None:
                self._add_confusion_matrix()

            if self.incorrect_test_images is not None:
                self._add_incorrect_test_images()

    def set_batch_test_loss(self, loss):
        self.batch_test_loss = loss

    def set_batch_test_accuracy(self, accuracy):
        self.batch_test_accuracy = accuracy

    # Maybe create a class for batch test data and for train test data?  To simplify complexity
    # of language and it'll simplify classes, too..
    def add_batch_test_confusion_data(self, predictions, labels):
        self.batch_test_predictions = predictions.cpu().numpy()
        self.batch_test_labels = labels.cpu().numpy()

    def _add_confusion_matrix(self):
        import sklearn.metrics
        cm = sklearn.metrics.confusion_matrix(self.batch_test_labels, self.batch_test_predictions)
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
        text = self._convert_text_to_markdown(str(text))
        self.writer.add_text(tag, text)

    def _convert_text_to_markdown(self, text):
        return text.replace("\n", "  \n")

    def add_incorrect_test_images(self, incorrect_flags, images):
        if (self.incorrect_test_images is not None and 
            len(self.incorrect_test_images) > self.max_num_incorrect_images): 
            return

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

