import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import random

'''settable parameters start'''
# raw-images paths
# if no testdir available set it to None like this: test_DATADIR = None
# if you have a dedicted one set a path to it just like with the one for trainig-data
train_DATADIR = r'.\AboveLayerShifting'
test_DATADIR = None

CATEGORIES = ['ClassA', 'ClassB']  # enter the names of the sub-directories for the classes (need to be exaclty the same for testdir if available)

# image adjustments
IMG_SIZE = 213  # enter the desired pixelvalue fro with and height (thesis dataset-default is 384)
IMG_DEPTH = 1  # enter image channels, 1=grayscale, 3=color
validation_split = 0.2  # validation split in case there is no test-set yet (0.0..1.0)
dataset_name = 'abovelayershifting'  # filename that goes in front of generated datasets
'''settable parameters end'''

training_data = []
if test_DATADIR is not None:
    test_data = []


def create_training_data():
    for category in CATEGORIES:

        path = os.path.join(train_DATADIR, category)  # create path to set CATEGORIES-strings
        class_num = CATEGORIES.index(category)  # get the classification by CATEGORIES-index

        for img in tqdm(os.listdir(path)):  # iterate over each image per A and B
            try:
                if IMG_DEPTH == 1:
                    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to gs array
                if IMG_DEPTH == 3:
                    img_array = cv2.imread(os.path.join(path, img))  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in case an image is faulty
                pass
        fig = plt.figure('sample for class {}'.format(category))
        # the print is here for google colab, as it cant show the title there
        print('sample for class {}'.format(category))
        plotimg = cv2.cvtColor(new_array.astype('uint8'), cv2.COLOR_BGR2RGB)
        plt.imshow(plotimg)
        plt.show()


def create_test_data():
    for category in CATEGORIES:

        path = os.path.join(test_DATADIR, category)
        class_num = CATEGORIES.index(category)

        for img in tqdm(os.listdir(path)):
            try:
                if IMG_DEPTH == 1:
                    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                if IMG_DEPTH == 3:
                    img_array = cv2.imread(os.path.join(path, img))
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                test_data.append([new_array, class_num])
            except Exception as e:
                pass
        fig = plt.figure('sample for class {}'.format(category))
        # the print is here for google colab, as it cant show the title there
        print('sample for class {}'.format(category))
        plotimg = cv2.cvtColor(new_array.astype('uint8'), cv2.COLOR_BGR2RGB)
        plt.imshow(plotimg)
        plt.show()


# MAKE TRAINSET
print('creating training data...')
create_training_data()

print('training-samples: ', (len(training_data)))  # validate samplecount

random.shuffle(training_data)  # randomize samples

print('verify shuffling of traindata:')
for sample in training_data[:10]:  # validate sample-shuffling
    print(sample[1])

# seperate images and labels
X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

# bring images in final numpy-shape
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, IMG_DEPTH)

# MAKE TESTSET
if test_DATADIR is not None:
    print('creating test data...')
    create_test_data()

    print('test-samples: ', len(test_data))

    random.shuffle(test_data)

    print('verify shuffling of testdata:')
    for sample in test_data[:10]:
        print(sample[1])

    X_test = []
    y_test = []

    for features, label in test_data:
        X_test.append(features)
        y_test.append(label)

    X_test = np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE, IMG_DEPTH)


# slices the training-dataset to get a fraction of it (size set by split) as
# testset
def slice_dataset(split=0.2):
    _x = len(X) * split
    _x = int(_x)
    xtest = X[-_x:]
    ytest = y[-_x:]
    xtrain = X[:(len(X)-_x)]
    ytrain = y[:(len(y) - _x)]

    return [xtrain, xtest, ytrain, ytest]


# slice is only executed if test_DATADIR is set to None in global parameters
if test_DATADIR is None:
    sets = slice_dataset(validation_split)

    X = sets[0]
    X_test = sets[1]
    y = sets[2]
    y_test = sets[3]
    print('sliced set, new values(X, X_test, y, y_test):', len(X), len(X_test), len(y), len(y_test))

# saving the sets for training and testing, format with used parameters for easy distinguishing
print('Saving data...')
np.save('{}_X_train_{}_{}.npy'.format(dataset_name, IMG_SIZE, IMG_DEPTH), X)
np.save('{}_y_train_{}_{}.npy'.format(dataset_name, IMG_SIZE, IMG_DEPTH), y)
np.save('{}_X_test_{}_{}.npy'.format(dataset_name, IMG_SIZE, IMG_DEPTH), X_test)
np.save('{}_y_test_{}_{}.npy'.format(dataset_name, IMG_SIZE, IMG_DEPTH), y_test)
print('Saved data!')
