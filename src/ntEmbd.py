import tensorflow as tf
import numpy as np
import argparse
import os
import sys
import random as python_random
from keras.models import Model
from textwrap import dedent
from time import strftime
from keras.callbacks import EarlyStopping, ModelCheckpoint
from matplotlib import pyplot


PYTHON_VERSION = sys.version_info
VERSION = "0.0.1"
PRORAM = "ntEmbd"
AUTHOR = "Saber Hafezqorani (UBC & BCGSC)"
CONTACT = "shafezqorani@bcgsc.ca"

def reset_seeds():
   np.random.seed(124)
   python_random.seed(124)
   tf.random.set_seed(124)

reset_seeds()

# Taken from https://github.com/lh3/readfq
def readfq(fp):  # this is a generator function
    last = None  # this is a buffer keeping the last unprocessed line
    while True:  # mimic closure; is it a bad idea?
        if not last:  # the first record or a record following a fastq
            for l in fp:  # search for the start of the next record
                if l[0] in '>@':  # fasta/q header line
                    last = l[:-1]  # save this line
                    break
        if not last:
            break
        name, seqs, last = last[1:], [], None
        for l in fp:  # read the sequence
            if l[0] in '@+>':
                last = l[:-1]
                break
            seqs.append(l[:-1])
        if not last or last[0] != '+':  # this is a fasta record
            yield name, ''.join(seqs), None  # yield a fasta record
            if not last:
                break
        else:  # this is a fastq record
            seq, leng, seqs = ''.join(seqs), 0, []
            for l in fp:  # read the quality
                seqs.append(l[:-1])
                leng += len(l) - 1
                if leng >= len(seq):  # have read enough quality
                    last = None
                    yield name, seq, ''.join(seqs)  # yield a fastq record
                    break
            if last:  # reach EOF before reading enough quality
                yield name, seq, None  # yield a fasta record instead
                break



def main():
    parser = argparse.ArgumentParser(
        description=dedent('''
        ntEmbd Pipeline
        -----------------------------------------------------------
        #Description of the whole ntEmbd pipeline
        '''),
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-v', '--version', action='version', version='ntEmbd ' + VERSION)
    subparsers = parser.add_subparsers(dest='mode', description=dedent('''
        You may run ntEmbd in several ways.
        For detailed usage of each mode:
            ntEmbd.py mode -h
        -------------------------------------------------------
        '''))
    parser_train = subparsers.add_parser('train', help="Run the ntEmbd on train mode")
    parser_train.add_argument('-a', '--train', help='Input sequences for training', required=True)
    parser_train.add_argument('-b', '--test', help='Input sequences for testing', required=True)
    parser_train.add_argument('-n', '--num_neurons', help='Number of neurons for the learned representation', default=64, type=int)
    parser_train.add_argument('-l', '--maxlen', help='The maximum length of input sequences', default=1000, type=int)
    parser_train.add_argument('-e', '--epoch', help='The number of epochs', default=100, type=int)
    parser_train.add_argument('-p', '--pad', help='Choose between "pre" and "post" padding', default="pre", type=str)
    parser_train.add_argument('-f', '--filename', help='Choose a prefix for model names', type=str)
    parser_train.add_argument('--no_mask_zero', help='Disable masking step for zero padded sequences', action='store_false', default=True)


    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    if args.mode == "train":
        train_numpy = args.train
        test_numpy = args.test
        num_nn = args.num_neurons
        epoch = args.epoch
        maxlen = args.maxlen
        mask = args.no_mask_zero
        pad = args.pad
        filename = args.filename

        print("\nrunning the ntEmbd (train mode) with following parameters\n")
        print("number of neurons", num_nn)
        print("max sequence length", maxlen)
        print("epoch", epoch)
        print("zero masking", mask)
        print("Padding", pad)

        full_name = filename + "_ntEmbd_notrun_" + pad + "pad_es_maxlen" + str(maxlen) + "_nn_" + str(num_nn) + "_mask_" + str(mask)

        X_train = np.load(train_numpy)
        X_test = np.load(test_numpy)


        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=5, output_dim=5, mask_zero=mask),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(num_nn, return_sequences=False), input_shape=(maxlen, 5)),
            tf.keras.layers.RepeatVector(maxlen),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='relu')),
        ])

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        mc = ModelCheckpoint("/projects/btl/shafez/projects/ANNote/analysis/lstm_autoencoder/ntEmbd_models/" + full_name + "_best", verbose=1, monitor='val_accuracy', mode='max', save_best_only=True)

        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

        history = model.fit(X_train, X_train, validation_split=0.2, epochs=epoch, callbacks=[es, mc])
        model.summary()

        layer_name = model.layers[1].name
        intermediate_layer_model = Model(inputs=model.input,
                                         outputs=model.get_layer(layer_name).output)

        layer_output_train = intermediate_layer_model.predict(X_train)
        layer_output_test = intermediate_layer_model.predict(X_test)


        model.save("/projects/btl/shafez/projects/ANNote/analysis/lstm_autoencoder/ntEmbd_models/" + full_name)
        np.savetxt("/projects/btl/shafez/projects/ANNote/analysis/lstm_autoencoder/embeddings/" + full_name + "_train", layer_output_train)
        np.savetxt("/projects/btl/shafez/projects/ANNote/analysis/lstm_autoencoder/embeddings/" + full_name + "_test", layer_output_test)
        sys.stdout.write(strftime("%Y-%m-%d %H:%M:%S") + ": Finished!\n")
        return



if __name__ == "__main__":
    main()
