# This file has some functions to construct a vocabulary
import tensorflow as tf
import collections


def read_words(filename):
    with tf.gfile.GFile(filename, "rb") as f:
        return f.read().decode("utf-8").replace("\n", "<eos>").split()


def build_vocab(filename):
    # build a vocabulary dictionary
    data = read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


def file_to_word_ids(filename, word_to_id):
    # turn a text file into a list of integers representing the words from the vocabulary
    data = read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]