#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob


def normalize(df_list):
    for i in range(len(df_list)):
        df_list[i]['KW'] = (
                    (df_list[i]['KW'] - df_list[i]['KW'].min()) / (df_list[i]['KW'].max() - df_list[i]['KW'].min()))

    return df_list


def get_data(pattern, columns):
    filenames = list(glob.glob(pattern))
    filenames.sort(key=str.lower)
    df_list = [pd.read_csv(filename, names=columns) for filename in filenames]
    return df_list, filenames


def save_output_image(fileName, plot):
    fileName = 'output/' + fileName
    plt.savefig(fileName)


def tanh(x):  # soft activation
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def test(array):
    for j in range(len(array)):
        x = j + 5
        y = array[j, 1]
        plt.scatter(x, y, color='red')
    plt.xlabel('Hour')
    plt.ylabel('normalized KW')
    plt.title("Testing Data")
    save_output_image('testingData', plt)
    # do stuff with the data
    # print('\tHour {}\t\tKW = {:6.3f}'.format(j + 5, array[j, 1]))


def makePower(dataset, degree):
    a = []
    for i in range(len(dataset)):
        z = 1
        for j in range(degree):
            z = z * dataset[i]
        a.append(z)
    return a


def train(dataFrame, datasetNumber):
    degree = 9
    dflength = len(dataFrame)
    Hours = np.transpose(dataFrame)[:-1]
    KiloWatts = np.transpose(dataFrame)[-1:]
    for i in range(dflength):
        X = int(round(Hours.iat[0, i], 0))
        Y = round(KiloWatts.iat[0, i], 3)
        X2 = np.power(X, 2)
        Y2 = np.power(Y, 2)

    SquareHours = makePower(Hours, 2)
    print(SquareHours)

    Hours = np.array(np.transpose(Hours)).flatten()
    KiloWatts = np.array(np.transpose(KiloWatts)).flatten()
    poly_fit = np.poly1d(np.polyfit(Hours, KiloWatts, degree))  # Illegal curve fit (Professor says "roll your own")
    xx = np.linspace(5, 20, 120)
    plt.plot(xx, poly_fit(xx), c='b', linestyle='-')
    for j in range(dataFrame.shape[0]):
        x = j + 5
        y = dataFrame.iat[j, 1]
        plt.scatter(x, y, color='red')
    plt.xlabel('Hour')
    plt.ylabel('normalized KW')
    plt.title("Training Data {}".format(datasetNumber))
    outputTitle = 'trainingdata{}'.format(datasetNumber)
    save_output_image(outputTitle, plt)
    plt.clf()


if __name__ == '__main__':
    col_names = ['Hour', 'KW']

    # read the datasets into memory
    train_DS, trainFiles = get_data('trainingFiles/train_data*.txt', col_names)
    test_DS, testFiles = get_data('testingFiles/test_data*.txt', col_names)

    for i in range(len(train_DS)):
        print("\n*************** Dataset {} *****************".format(i))
        train(train_DS[i], i + 1)
        plt.show()
        plt.close()

    # for each test set
    for i in range(len(test_DS)):
        test_DS[i] = np.asarray(test_DS[i])
        print("\n*** Testing *** with file at path {}  \n\t\tImage saved to output folder.".format(testFiles[i]))
        test(test_DS[i])
        plt.close()