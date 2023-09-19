import argparse
import os
import sys
import random
from sklearn.model_selection import train_test_split, KFold
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, RepeatVector, TimeDistributed, Dropout
import tensorflow as tf
import optuna
from collections import Counter
import matplotlib.pyplot as plt


PYTHON_VERSION = sys.version_info
VERSION = "0.9"
PRORAM = "ntEmbd"
AUTHOR = "Saber Hafezqorani (UBC & BCGSC)"
CONTACT = "shafezqorani@bcgsc.ca"

# A function to set the random seed for reproducibility
def reset_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

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

# Angular distance of two unit vectors with positive values - this one is experimental (for testing)
def angular_distance(vector_a, vector_b):
    """
    Calculate the angular distance between two vectors.

    Parameters:
    - vector_a, vector_b: Numpy arrays representing the two vectors.

    Returns:
    - Angular distance between the two vectors.
    """

    # Calculate the dot product of the two vectors
    dot_product = np.dot(vector_a, vector_b)

    # Calculate the magnitudes (norms) of the two vectors
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    # Calculate the cosine of the angle between the two vectors
    cosine_angle = dot_product / (norm_a * norm_b)

    # Ensure the value lies in the domain of arccos (-1 to 1) due to potential floating-point inaccuracies
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    # Calculate the angular distance
    angle = np.arccos(cosine_angle)

    return angle

# Angular distance of two unit vectors with positive values - using TensorFlow operations
def angular_distance_tf(vector_a, vector_b):
    """
    Calculate the angular distance between two vectors using TensorFlow operations.

    Parameters:
    - vector_a, vector_b: TensorFlow tensors representing the two vectors.

    Returns:
    - Angular distance between the two vectors.
    """

    # Ensure the vectors are of type float32
    vector_a = tf.cast(vector_a, tf.float32)
    vector_b = tf.cast(vector_b, tf.float32)

    # Calculate the dot product of the two vectors
    dot_product = tf.reduce_sum(tf.multiply(vector_a, vector_b))

    # Calculate the magnitudes (norms) of the two vectors
    norm_a = tf.norm(vector_a)
    norm_b = tf.norm(vector_b)

    # Calculate the cosine of the angle between the two vectors
    cosine_angle = dot_product / (norm_a * norm_b)

    # Ensure the value lies in the domain of arccos (-1 to 1) due to potential floating-point inaccuracies
    cosine_angle = tf.clip_by_value(cosine_angle, -1.0, 1.0)

    # Calculate the angular distance
    angle = tf.acos(cosine_angle)

    return angle

# IUPAC encoding for nucleotide sequences
def iupac_encoding(seq):
    """
    One-hot encode an RNA sequence using IUPAC symbols in a binary format.

    Parameters:
    - sequence: RNA sequence string

    Returns:
    - Numpy array with one-hot encoded sequence
    """

    # Define the 4-bit encoding dictionary for IUPAC symbols
    encoding_dict = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1],
        'U': [0, 0, 0, 1],
        'W': [1, 0, 0, 1],
        'S': [0, 1, 1, 0],
        'M': [1, 1, 0, 0],
        'K': [0, 0, 1, 1],
        'R': [1, 0, 1, 0],
        'Y': [0, 1, 0, 1],
        'B': [0, 1, 1, 1],
        'D': [1, 0, 1, 1],
        'H': [1, 1, 0, 1],
        'V': [1, 1, 1, 0],
        'N': [1, 1, 1, 1],
        'Z': [0, 0, 0, 0]
    }

    # Encode the sequence using the dictionary
    encoded_sequence = [encoding_dict[base] for base in seq]
    return np.array(encoded_sequence)

# pre-process sequences for training
def process_sequences(sequences, max_length, truncate_long_sequences, pad_position, padding_value=(-1, -1, -1, -1)):
    """
    Process raw sequences: filter based on length, encode, and then pad/truncate.

    Parameters:
    - sequences: List of raw sequences (strings)
    - max_length: The desired length for each sequence after padding/truncating
    - truncate_long_sequences: Whether to truncate sequences longer than max_length
    - padding_value: The value used for padding (default is [-1, -1, -1, -1])
    - pad_position: Whether to pad/truncate at the start ("pre") or end ("post") of the sequence

    Returns:
    - List of processed sequences
    """

    processed_sequences = []

    for seq in sequences:
        if len(seq) > max_length:
            if truncate_long_sequences == "truncate_end":
                seq = seq[:max_length]
            elif truncate_long_sequences == "truncate_start":
                seq = seq[-max_length:]
            else:
                continue  # Skip truncation and ignore this sequence
        else:
            if pad_position not in ["pre", "post"]:
                continue # Skip padding and ignore this sequence
        
        # Encode the sequence
        encoded_seq = iupac_encoding(seq)
        
        # Pad the encoded sequence
        if len(encoded_seq) < max_length:
            pad_length = max_length - len(encoded_seq)
            padding = np.array([list(padding_value)] * pad_length)

            if pad_position == "pre":
                encoded_seq = np.vstack((padding, encoded_seq))
            else:  # "post"
                encoded_seq = np.vstack((encoded_seq, padding))
        
        processed_sequences.append(encoded_seq)

    return processed_sequences

# Build a Bi-LSTM autoencoder
def build_bilstm_autoencoder(seq_len, embedding_size, feature_dim, lstm_units, dropout_rate, activation):
    """
    Build a Bi-LSTM autoencoder.
    """

    lstm_units_2 = lstm_units // 2

    # Encoder
    inputs = Input(shape=(seq_len, feature_dim))
    encoded = Bidirectional(LSTM(lstm_units, return_sequences=True, activation=activation))(inputs)
    encoded = Dropout(dropout_rate)(encoded)
    encoded = Bidirectional(LSTM(lstm_units_2, return_sequences=False, activation=activation))(encoded)
    encoded_latent = Dense(embedding_size, activation=activation)(encoded)
    
    # Decoder
    decoded = RepeatVector(seq_len)(encoded_latent) # Convert 2D latent representation to 3D
    decoded = LSTM(lstm_units_2, return_sequences=True, activation=activation)(decoded)
    decoded = Dropout(dropout_rate)(decoded)
    decoded = LSTM(lstm_units, return_sequences=True, activation=activation)(decoded)
    decoded = TimeDistributed(Dense(feature_dim, activation='softmax'))(decoded)
    
    # Autoencoder
    autoencoder = Model(inputs, decoded)

    # Embedding model
    embedding_model = Model(inputs, encoded_latent)

    return autoencoder, embedding_model

# Define the objective function for Optuna optimization
def optuna_objective(max_length, architecture, epoch, trial, X_train, X_val):
    # Hyperparameters to be tuned

    # Learning rate
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

    # Batch size
    #batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    batch_size = trial.suggest_categorical("batch_size", [16, 32])

    # Number of units in the LSTM layer
    #lstm_size = trial.suggest_categorical('units', [256, 512, 1024])
    lstm_size = trial.suggest_categorical('units', [32, 64, 128])

    # Latent dimension (embedding size)
    #embedding_size = trial.suggest_categorical("latent_dim", [128, 256, 512])
    embedding_size = trial.suggest_categorical("latent_dim", [20, 30, 40])

    # Optimizer choice
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd"])
    if optimizer_name == "adam":
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr)
    elif optimizer_name == "sgd":
        optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=lr)

    # Dropout rate for regularization
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5, step=0.1)

    # Activation function choice
    activation = trial.suggest_categorical("activation", ["relu", "tanh", "sigmoid"])

    # Build and compile the autoencoder
    if architecture == 'bilstm':
        
        autoencoder, embedding_model = build_bilstm_autoencoder(max_length, embedding_size, 4, lstm_size, dropout_rate, activation)
        
        # Train the model
        autoencoder.compile(optimizer=optimizer, loss=angular_distance_tf)
        autoencoder.fit(X_train, X_train, epochs=epoch, batch_size=batch_size, shuffle=True, validation_data=(X_val, X_val), verbose=0)

        # Return validation loss
        val_loss = autoencoder.evaluate(X_val, X_val, verbose=0)
        return val_loss

    elif architecture == 'transformer':
        print("Transformer model is not implemented yet.")
        sys.exit(1)

# Aggregate the optuna trials and yeild the best set of hyperparameters
def aggregate_hyperparameters(best_hyperparameters):
    """
    Aggregate hyperparameters across folds using the Voting/Most Frequent strategy
    for categorical parameters and Averaging for continuous parameters.
    
    Args:
    - best_hyperparameters (list): List of dictionaries. Each dictionary contains 
      the best hyperparameters for a fold.
      
    Returns:
    - aggregated_params (dict): Dictionary containing the aggregated hyperparameters.
    """
    
    # Initialize aggregated_params dictionary
    aggregated_params = {}
    
    # For each hyperparameter, check if it's categorical or continuous and aggregate accordingly
    for key in best_hyperparameters[0].keys():
        if key == "lr":
            # Average the learning rates
            aggregated_params[key] = sum([params[key] for params in best_hyperparameters]) / len(best_hyperparameters)
        else:
            # Use the Voting/Most Frequent strategy for other hyperparameters
            most_common = Counter([params[key] for params in best_hyperparameters]).most_common(1)
            aggregated_params[key] = most_common[0][0]
    
    return aggregated_params


    """
    Plot the training and validation metrics from the Keras history object and save the plots.
    
    Args:
    - history (History): Keras History object returned by the .fit() method.
    - save_dir (str): Directory to save the plots.
    
    Returns:
    - None (Displays and saves the plots).
    """
    
    # Extract loss from the history object
    loss = history.history['loss']
    epochs = range(1, len(loss) + 1)
    
    # Plot training loss
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, loss, 'bo', label='Training loss')
    
    # Plot validation loss if available
    if 'val_loss' in history.history:
        val_loss = history.history['val_loss']
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'training_validation_loss.png'))
    plt.show()
    
    # If there are other metrics in the history object, plot them
    for metric, values in history.history.items():
        if metric not in ['loss', 'val_loss']:
            plt.figure(figsize=(12, 6))
            plt.plot(epochs, values, 'bo', label=f'Training {metric}')
            
            # Plot validation metric if available
            if f'val_{metric}' in history.history:
                val_values = history.history[f'val_{metric}']
                plt.plot(epochs, val_values, 'b', label=f'Validation {metric}')
            
            plt.title(f'Training and Validation {metric}')
            plt.xlabel('Epochs')
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.savefig(os.path.join(save_dir, f'training_validation_{metric}.png'))
            plt.show()

# Plot the training history and save the plots
def plot_and_save_training_history(history, save_dir):
    """
    Plot the training and validation metrics from the Keras history object and save the plots.
    
    Args:
    - history (History): Keras History object returned by the .fit() method.
    - save_dir (str): Directory to save the plots.
    
    Returns:
    - None (Displays and saves the plots).
    """
    
    # Extract loss from the history object
    loss = history.history['loss']
    epochs = range(1, len(loss) + 1)
    
    # Plot training loss
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, loss, 'bo', label='Training loss')
    
    # Plot validation loss if available
    if 'val_loss' in history.history:
        val_loss = history.history['val_loss']
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'training_validation_loss.png'))
    plt.show()
    
    # If there are other metrics in the history object, plot them
    for metric, values in history.history.items():
        if metric not in ['loss', 'val_loss']:
            plt.figure(figsize=(12, 6))
            plt.plot(epochs, values, 'bo', label=f'Training {metric}')
            
            # Plot validation metric if available
            if f'val_{metric}' in history.history:
                val_values = history.history[f'val_{metric}']
                plt.plot(epochs, val_values, 'b', label=f'Validation {metric}')
            
            plt.title(f'Training and Validation {metric}')
            plt.xlabel('Epochs')
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.savefig(os.path.join(save_dir, f'training_validation_{metric}.png'))
            plt.show()

# Analyze RNA sequences and plot their length distribution
def analyze_sequences(sequences, max_length):
    """
    Perform statistical analysis on RNA sequences and plot their length distribution.
    
    Args:
    - sequences (list): List of RNA sequences.
    - max_length (int): The maximum sequence length to be used for the vertical line.
    
    Returns:
    - None (Displays the statistics and plots).
    """
    
    # Calculate sequence lengths
    sequence_lengths = [len(seq) for seq in sequences]
    
    # Basic statistics
    total_sequences = len(sequences)
    min_seq_length = min(sequence_lengths)
    max_seq_length = max(sequence_lengths)
    mean_seq_length = np.mean(sequence_lengths)
    median_seq_length = np.median(sequence_lengths)
    
    # Display statistics
    print(f"Total RNA sequences: {total_sequences}")
    print(f"Shortest sequence length: {min_seq_length}")
    print(f"Longest sequence length: {max_seq_length}")
    print(f"Mean sequence length: {mean_seq_length:.2f}")
    print(f"Median sequence length: {median_seq_length}")
    print(f"Percentage of sequences longer than {max_length}: {sum(1 for x in sequence_lengths if x > max_length) / total_sequences * 100:.2f}%")
    
    # Plotting the length distribution
    plt.figure(figsize=(10, 6))
    plt.hist(sequence_lengths, bins=np.logspace(np.log10(min_seq_length), np.log10(max_seq_length), 50), weights=np.ones(total_sequences) / total_sequences)
    plt.gca().set_xscale("log")
    plt.gca().set_ylim(0, 1)
    plt.axvline(x=max_length, color='red', linestyle='--', label=f"Max length: {max_length}")
    plt.title("RNA Sequence Length Distribution")
    plt.xlabel("Sequence Length (Log Scale)")
    plt.ylabel("Percentage of Sequences")
    
    # Set x-axis ticks to be 10 to the power of 1, 2, 3, etc.
    max_power = int(np.ceil(np.log10(max_seq_length)))
    plt.xticks([10**i for i in range(1, max_power+1)], [f"$10^{i}$" for i in range(1, max_power+1)])
    
    plt.legend()
    plt.grid(True, which="both", ls="--", c='0.65')
    plt.show()

# Main function
def main():
    parser = argparse.ArgumentParser(description='ntEmbd: Deep learning embedding for nucleotide sequences')
    subparsers = parser.add_subparsers(dest='mode', required=True, help='Sub-command help')

    # Train subparser
    train_parser = subparsers.add_parser('train', help='Train a new model.')

    # Required arguments
    train_parser.add_argument('input_fasta', type=str, nargs='+', help='Path(s) to the input FASTA file(s) for training. You can provide multiple paths separated by spaces.')

    # Optional arguments
    train_parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training the model.')
    train_parser.add_argument('--optuna_epoch', type=int, default=5, help='Number of epochs for training the model during hyperparameter optimization.')
    train_parser.add_argument('--embedding_size', type=int, default=128, help='Size of the latent representation/embedding.')
    train_parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    train_parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    train_parser.add_argument('--save_model', type=str, default=None, help='Path to save the trained model.')
    train_parser.add_argument('--load_model', type=str, default=None, help='Path to a pre-trained model to continue training or for embedding generation.')
    train_parser.add_argument('--save_embeddings', action='store_true', help='Generate embeddings for the training data and save them.')
    train_parser.add_argument('--log_dir', type=str, default='./logs', help='Directory to save training logs and TensorBoard data.')
    train_parser.add_argument('--gpu', action='store_true', help='Use GPU for training if available.')
    train_parser.add_argument('--seed', type=int, default=192, help='Random seed for reproducibility.')
    train_parser.add_argument('--padding', type=str, choices=['pre', 'post', 'ignore'], default='post', help='Choose the padding position: "pre" for start and "post" for end.')
    train_parser.add_argument('--max_length', type=int, default=1000, help='Maximum length of sequences to be considered. Default is 1000 base pairs.')
    train_parser.add_argument('--long_seq', type=str, choices=['truncate_start', 'truncate_end', 'ignore'], default='ignore', help='How to handle sequences longer than max_length: "truncate" or "ignore". Default is "truncate".')
    train_parser.add_argument('--arch', choices=['bilstm', 'transformer'], default='bilstm', help='Model architecture (default: bilstm)')
    train_parser.add_argument('--loss', choices=['angular_distance', 'mse'], default='angular_distance', help='Loss function (default: angular_distance)')
    train_parser.add_argument('--optimizer', choices=['adam', 'sgd'], default='adam', help='Optimizer (default: adam)')
    train_parser.add_argument('--early_stopping', action='store_true', default=False, help='Enable early stopping.')
    train_parser.add_argument('--hyperparameter_optimization', action='store_true', default=False, help='Enable hyperparameter optimization.')
    train_parser.add_argument('--optuna_trial', type=int, default=10, help='Number of trials for hyperparameter optimization using Optuna.')
    train_parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate for regularization.')
    train_parser.add_argument('--lstm_units', type=int, default=256, help='Number of units in the LSTM layer.')
    train_parser.add_argument('--activation', choices=['relu', 'tanh', 'sigmoid'], default='relu', help='Activation function (default: relu)')

    # Embed subparser
    embed_parser = subparsers.add_parser('embed', help='Generate embeddings using a pre-trained model.')
    embed_parser.add_argument('input_fasta', type=str, help='Path to the input FASTA file for generating embeddings.')
    embed_parser.add_argument('model_path', type=str, help='Path to the pre-trained model.')
    # ... other embedding related args ...

    # Evaluate subparser
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model.')
    eval_parser.add_argument('input_fasta', type=str, help='Path to the input FASTA file for evaluation.')
    eval_parser.add_argument('model_path', type=str, help='Path to the trained model.')
    # ... other evaluation related args ...

    # Fine-tune subparser
    finetune_parser = subparsers.add_parser('fine-tune', help='Fine-tune a pre-trained model on a new dataset.')
    finetune_parser.add_argument('input_fasta', type=str, help='Path to the input FASTA file for fine-tuning.')
    finetune_parser.add_argument('model_path', type=str, help='Path to the pre-trained model to continue training.')
    # ... other fine-tuning related args ...

    # Visualize subparser
    visualize_parser = subparsers.add_parser('visualize', help='Visualize the embeddings.')
    visualize_parser.add_argument('embeddings_path', type=str, help='Path to the embeddings file.')
    # ... other visualization related args ...

    # Info subparser
    info_parser = subparsers.add_parser('info', help='Get information about a trained model.')
    info_parser.add_argument('model_path', type=str, help='Path to the trained model.')

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    # Depending on the sub-command, call the respective functions
    if args.mode == 'train':

        reset_seeds(args.seed)

        all_sequences = []
        for fasta_file in args.input_fasta:
            with open(fasta_file, 'rt') as f:
                for seqN, seqS, seqQ in readfq(f):
                    all_sequences.append(seqS)

        # Analyze the sequences and plot their length distribution
        analyze_sequences(all_sequences, args.max_length)

        # pre-process sequences for training
        processed_sequences = process_sequences(all_sequences, args.max_length, args.long_seq, args.padding, padding_value=(-1, -1, -1, -1))
        processed_sequences_array = np.array(processed_sequences)

        #Splitting the data and setting up cross-validation
        train_val_data, test_data = train_test_split(processed_sequences_array, test_size=0.20, random_state=args.seed)
        train_data, val_data = train_test_split(train_val_data, test_size=0.25, random_state=args.seed)  # 0.25 x 0.80 = 0.20

        # Save the train, validation, and test data. Get directory from args.save_model if provided, otherwise use the path of the FASTA file
        if args.save_model:
            save_dir = os.path.dirname(args.save_model)
        else:
            save_dir = os.path.dirname(args.input_fasta[0])

        np.save(save_dir + "train_data.npy", train_data)
        np.save(save_dir + "val_data.npy", val_data)
        np.save(save_dir + "test_data.npy", test_data)

        kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)  # Using 5-fold cross-validation

        # Check if hyperparameter optimization is enabled or not and set the number of trials for Optuna
        if args.hyperparameter_optimization:

            # Sample 1/10 of the data for hyperparameter tuning
            sample_indices = np.random.choice(train_data.shape[0], size=int(train_data.shape[0] * 0.1), replace=False)
            sampled_data = train_data[sample_indices]

            # Store validation losses and best hyperparameters
            validation_losses = []
            best_hyperparameters = []
            n_trials = args.optuna_trial
            for fold_num, (train_index, val_index) in enumerate(kf.split(sampled_data), start=1):  # Using start=1 to begin counting from 1
                print(f"Processing Fold {fold_num} ...")
                # Split data into training and validation sets
                X_train, X_val = sampled_data[train_index], sampled_data[val_index]

                # Initialize Optuna study
                study = optuna.create_study(direction="minimize")
                study.optimize(lambda trial: optuna_objective(args.max_length, args.arch, args.optuna_epoch, trial, X_train, X_val), n_trials=n_trials)

                # Append best loss and hyperparameters for this fold
                validation_losses.append(study.best_value)
                best_hyperparameters.append(study.best_params)
            
            # Compute average and standard deviation of validation losses
            average_loss = np.mean(validation_losses)
            std_loss = np.std(validation_losses)
            print(f"Average validation loss: {average_loss:.4f} ± {std_loss:.4f}")

            # aggregate the hyperparameters across folds
            best_hyperparameters = aggregate_hyperparameters(best_hyperparameters)
            print(f"Best hyperparameters: {best_hyperparameters}")
            embedding_size = best_hyperparameters["latent_dim"]
            dropout_rate = best_hyperparameters["dropout_rate"]
            lstm_units = best_hyperparameters["units"]
            activation = best_hyperparameters["activation"]
            optimizer = best_hyperparameters["optimizer"]
            batch_size = best_hyperparameters["batch_size"]
            learning_rate = best_hyperparameters["lr"] # Learning rate is optimized within the optimizer. So, we don't need to use it here.

        else:
            # Instead of Hyperparameter tuning, use the parameters provided (or default values)
            embedding_size = args.embedding_size
            dropout_rate = args.dropout_rate
            lstm_units = args.lstm_units
            activation = args.activation
            optimizer_choice = args.optimizer
            if optimizer_choice == "adam":
                optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
            elif optimizer_choice == "sgd":
                optimizer = tf.keras.optimizers.SGD(learning_rate=args.learning_rate)
            batch_size = args.batch_size

        log_dir = args.log_dir
        epoch = args.epochs
        loss = args.loss
        if loss == 'angular_distance':
            loss = angular_distance_tf
            
        # Build and compile the model (either with optimized or default hyperparameters)
        autoencoder, embedding_model = build_bilstm_autoencoder(args.max_length, embedding_size, 4, lstm_units, dropout_rate, activation)
        autoencoder.compile(optimizer=optimizer, loss=loss)
        autoencoder.summary()

        # Write the model summary to a file in the log directory
        with open(log_dir + "model_summary.txt", "w") as f:
            autoencoder.summary(print_fn=lambda x: f.write(x + '\n'))
        
        # Train the model using the whole training set and validate using the separate validation set
        train_data = tf.convert_to_tensor(train_data, dtype=tf.float32)
        val_data = tf.convert_to_tensor(val_data, dtype=tf.float32)
        
        # Introduce early stopping and model checkpoints
        if args.early_stopping:
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            model_checkpoint = tf.keras.callbacks.ModelCheckpoint(save_dir + 'best_model.h5', monitor='val_loss', save_best_only=True)
            history = autoencoder.fit(train_data, train_data, epochs=epoch, batch_size=batch_size, shuffle=True, validation_data=(val_data, val_data), callbacks=[early_stopping, model_checkpoint])
        else:
            history = autoencoder.fit(train_data, train_data, epochs=epoch, batch_size=batch_size, shuffle=True, validation_data=(val_data, val_data))
        
        # Plot the training history and save it to a file in the log directory
        with open(log_dir + "training_history.txt", "w") as f:
            f.write(str(history.history))

        plot_and_save_training_history(history, save_dir)
        
        # Compute validation loss and print it
        val_loss = autoencoder.evaluate(val_data, val_data, verbose=0)
        print(f"Validation loss: {val_loss:.4f}")

        # Save the embeddings if --save_embeddings is enabled
        if args.save_embeddings:
            # Generate embeddings for the training data
            train_embeddings = embedding_model.predict(train_data)
            np.savetxt(save_dir + "train_embeddings.tsv", train_embeddings, delimiter='\t')
            
    elif args.mode == 'embed':
        # call embedding function
        pass
    # ... handle other sub-commands ...


if __name__ == "__main__":
    main()
