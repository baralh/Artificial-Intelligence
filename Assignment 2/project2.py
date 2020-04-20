#!/usr/bin/env python
import glob
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _reindex_dataframe(dataframe):
    """Helper method that re-indexes a DataFrame object.
    Args:
        dataframe (DataFrame): The DataFrame to re-index.
    Returns:
        The re-indexed DataFrame.
    """
    return dataframe.reset_index(drop=True)


class perceptron():
    """A Class that represents a perceptron-based classifier.
    Attributes:
        training_data (DataFrame): The data used for training.
        knownClass (DataFrame): The desired output.
        weights (list): The list of weights.
        alpha (float): The learning constant.
        numcycles (int): The number of training cycles (upper limit)
        epsilon: The minimum total error to reach.
        total_error (float): The total error of the perceptron.
    """

    def __init__(self, weights):
        self.weights = weights

    def train(self,
              training_data,
              known_class,
              alpha,
              num_cycles,
              epsilon,
              hard=True):
        """Trains a perceptron.
        Args:
            training_data (DataFrame): The training data to use.
            known_class (DataFrame): The desired output.
            alpha: The learning constant.
            num_cycles (int): The number of training iterations.
            epsilon: The minimum total error to achieve.
            hard (bool): Whether to use hard or soft activation. Default = True.
        Returns:
            dict: The final 'cycle', 'weights', and 'total error'.
        """
        self.training_data = _reindex_dataframe(training_data)
        self.known_class = _reindex_dataframe(known_class)
        self.alpha = alpha
        self.num_cycles = num_cycles
        self.epsilon = epsilon
        weights = self.weights[:]  # Holds current weights.
        self.weights = []
        self.weights.append(weights[:])  # Holds all weights.
        result = {'cycle': None, 'weights': None, 'total error': None}

        for cycle in range(self.num_cycles):  # Iterates for all cycles.
            self.total_error = 0.0

            # Iterates for all patterns.
            for row in range(len(self.training_data)):
                net = 0
                predicted_class = None

                # Iterates for all inputs.
                for feature in range(len(self.training_data.columns)):
                    net += (weights[feature] *
                            self.training_data.iloc[row, feature])
                net += weights[len(self.training_data.columns)]  # Adds bias

                if hard:
                    predicted_class = self._hard_activation(net)
                else:
                    predicted_class = self._soft_activation(net)

                error = self.known_class[row] - predicted_class
                self.total_error += error**2
                learn = self.alpha * error

                # Updates weights
                for feature in range(len(self.training_data.columns)):
                    weights[feature] += (
                        learn * self.training_data.iloc[row, feature])
                weights[len(self.training_data.columns)] += learn
                """print('Cycle: {}\t\tRow: {}\t\tAlpha: {}\t\tError: {}'.format(
                    cycle + 1, row + 1, alpha, error))"""

            self.weights.append(weights[:])

            if self.total_error <= epsilon:
                """
                print('Total Error less than epsilon achieved in cycle #{}'
                      '\nPerceptron weights will be set to {}'
                      '\nTotal Error for Perceptron was computed to be {}\n'
                      '**************************************'.format(
                          cycle + 1, self.weights[:], self.total_error))
                """
                result['cycle'] = cycle + 1
                break

        result['weights'] = weights[:]
        result['total error'] = self.total_error
        return result

    def test(self):
        pass

    def _hard_activation(self, x):
        """Performs hard activation."""
        prediction = np.sign(x)
        if prediction >= 0:
            return 1
        else:
            return 0

    def _soft_activation(self, x, k=5):
        """Performs the unipolar hyperbolic tangent function.
        Args:
            x: The value to perform the function on.
            k (int): The gain.
        """
        out = 1 / (1 + np.exp(-k * x))
        return out


def partition(dataset, percentage):
    """Splits the dataset into training and testing sets.
    Args:
        dataset (DataFrame): The data to split into parts.
        percentage (int): The percentage of data to be used in the training set.
    """
    # Shuffles dataset.
    dataset = dataset.iloc[np.random.permutation(len(dataset))]
    dataset = dataset.reset_index(drop=True)

    train_data = dataset.copy()
    # Creates empty DataFrame with labeled columns.
    test_data = pd.DataFrame(data=None, columns=dataset.columns)

    # Sets integer amount representing percentage of data to use for testing.
    test_range = int(len(dataset) * ((100 - percentage) / 100.0))
    for _ in range(test_range):
        rand = random.randrange(len(train_data))
        row = train_data.iloc[rand]
        test_data = test_data.append(row)

        # Removes corresponding test datum from training set.
        train_data.drop(train_data.index[rand], inplace=True)

    return train_data, test_data


def init_weights(num, lo=-0.4, hi=0.4):
    """Sets up random initial weights between a range for the classifier.
    Args:
        num (int): The number of weights to create.
        lo: The lowest weight value (inclusive). Default is -0.4.
        hi: The highest weight value (inclusive). Default is 0.4.
    Returns:
        list: The list of random weights.
    """
    weights = []
    for _ in range(num):
        weights.append(random.randint(lo * 10, hi * 10) / 10.0)

    return weights


def get_data(pattern, columns):
    """Retrieves data from files whose names match the pattern.
    Args:
        pattern (str): The pattern to match for.
        columns (list): A list of strings used for column labels.
    Returns:
        A list of DataFrames and a list of filenames.
    """
    filenames = list(glob.glob(pattern))
    filenames.sort(key=str.lower)
    df_list = [pd.read_csv(filename, names=columns) for filename in filenames]
    return df_list, filenames


def normalize_zscore(df_list, cols_to_norm):
    """Performs z-score normalization on a list of DataFrame lists.
    Args:
        df_list: List of DataFrame lists to be normalized.
        cols_to_norm (list): The list of columns to normalize.
    Returns:
        List of normalized DataFrame objects.
    """
    for i in range(len(df_list)):
        df_list[i][cols_to_norm] = (
            df_list[i][cols_to_norm] -
            df_list[i][cols_to_norm].mean()) / df_list[i][cols_to_norm].std()
        df_list[i] = df_list[i].dropna()

    return df_list


def show_scatter(fig):
    """Creates temp figure manager to display desired figure."""
    temp = plt.figure()
    new_manager = temp.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)
    plt.show()


def create_scatter(df, title=None):
    """Plots DataFrame data on a scatter plot.
    Args:
        df (DataFrame): The DataFrame to plot.
        title (str): The title of the graph.
    """
    fig, ax = plt.subplots()

    if title is not None:
        plt.title(title)

    cols = [col for col in df.columns]
    ax.scatter(x=df[cols[0]], y=df[cols[1]], s=1, c=df[cols[2]])
    return fig, ax


def main():
    col_names = ['Height', 'Weight', 'Class']
    cols_to_norm = ['Height', 'Weight']
    filename = 'group*.txt'
    alpha = 5
    cycles = 5000
    percentage = [75, 25]
    epsilon = 0
    fmt = '.png'
    weights = init_weights(len(cols_to_norm) + 1)
    results = []

    print('Compiling Data...', end='', flush=True)
    df_list, filenames = get_data(filename, col_names)
    num_files = len(filenames)
    print('\t\t\t\tDone.')

    print('Normalizing data...', end='', flush=True)
    df_list = normalize_zscore(df_list, cols_to_norm)
    print('\t\t\t\tDone.')

    print('Plotting Data Set...', end='', flush=True)
    # Plots data set.
    for i in range(num_files):
        group, _ = os.path.splitext(filenames[i])
        df = df_list[i]
        title = 'Data Set ' + group
        fig, _ = create_scatter(df, title)
        #plt.show()
        fig.savefig(title + fmt)
    print('\t\t\t\tDone.')

    print('Creating Training Set and Testing Set...', end='', flush=True)
    # Splits data set into training set and testing set based on percentage.
    train_list = []
    test_list = []

    for i in range(num_files):
        for j in range(len(percentage)):
            train_data, test_data = partition(df_list[i], percentage[j])
            train_list.append(train_data)
            test_list.append(test_data)
    print('\tDone.')

    print('Plotting Training Set and Testing Set...', end='', flush=True)
    # Plots training data set and testing data set separately.
    for i in range(num_files):
        for j in range(len(percentage)):
            group, _ = os.path.splitext(filenames[i])

            train_data = train_list[(2 * i) + j]
            title = 'Training Set {} Pr.2.1.{}'.format(group, j + 1)
            fig, _ = create_scatter(train_data, title)
            #plt.show()
            fig.savefig(title + fmt)

            test_data = test_list[(2 * i) + j]
            title = 'Testing Set {} Pr.2.1.{}'.format(group, j + 1)
            fig, _ = create_scatter(test_data, title)
            #plt.show()
            fig.savefig(title + fmt)
    print('\tDone.')

    print('Training Perceptron using hard activation...', end='', flush=True)
    # Runs through each data set and trains the perceptron using a hard
    # activation function.
    for i in range(num_files):
        # Sets epsilons for Datasets A, B, and C respectively.
        if i == 0:
            epsilon = 10**(-5)
        elif i == 1:
            epsilon = 100
        else:
            epsilon = 1.45 * (10**3)

        for j in range(len(percentage)):
            me = perceptron(weights)

            # Trains the perceptron using features and resulting observations.
            result = me.train(
                train_list[(2 * i) + j][cols_to_norm],
                train_list[(2 * i) + j]['Class'],
                alpha,
                cycles,
                epsilon,
                hard=True)

            result = list(result.values())
            result.append(filenames[i])
            result.append(percentage[j])
            result.append('unipolar hard activation')
            results.append(result)
    print('\tDone.')

    print('Training Perceptron using soft activation...', end='', flush=True)
    # Runs through each data set and trains the perceptron using a soft
    # activation function.
    for i in range(num_files):
        # Sets epsilons for Datasets A, B, and C, respectively.
        if i == 0:
            epsilon = 10**(-5)
        elif i == 1:
            epsilon = 100
        else:
            epsilon = 1.45 * (10**3)

        for j in range(len(percentage)):
            me = perceptron(weights)

            # Trains the perceptron using features and resulting observations.
            result = me.train(
                train_list[(2 * i) + j][cols_to_norm],
                train_list[(2 * i) + j]['Class'],
                alpha,
                cycles,
                epsilon,
                hard=False)

            result = list(result.values())
            result.append(filenames[i])
            result.append(percentage[j])
            result.append('unipolar hyperbolic tangent')
            results.append(result)
    print('\tDone.')

    print('\n********************** RESULTS **********************')

    for x in results:
        space = '  \t' if len(str(x[2])) >= 10 else '\t\t\t'
        print(
            'Filename: {}\t\tTraining Percentage: {}\t\tFunction: {}\tCycle: {}'
            .format(x[3], x[4], x[5], x[0]))
        print('Total Error: {}{}Weights: {}\n'.format(x[2], space, x[1]))


if __name__ == '__main__':
    main()