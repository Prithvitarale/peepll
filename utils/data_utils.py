import numpy as np


# Function Description:
# Takes data and classes, and returns data of those classes alone.
def datafilter(X_train, Y_train, X_test, Y_test, classes):
    if np.shape(Y_train)[1] == 1:
        train_mask = np.isin(Y_train[:, 0], classes)
        test_mask = np.isin(Y_test[:, 0], classes)
    elif np.shape(Y_train)[1] > 1:
        train_mask = np.isin(Y_train.argmax(1), classes)
        test_mask = np.isin(Y_test.argmax(1), classes)
    else:
        train_mask = np.isin(Y_train, classes)
        test_mask = np.isin(Y_test, classes)
    print("datafilter classes", classes)
    X_train, Y_train = X_train[train_mask], Y_train[train_mask]
    X_test, Y_test = X_test[test_mask], Y_test[test_mask]
    return (X_train, Y_train, X_test, Y_test)

# Function Description:
# Takes data and classes, and returns data of those classes in blocks of maximum length num_per_class.
def datafilter_perclass(X_train, Y_train, num_per_class, classes, negate=False):

    if np.shape(Y_train)[1] == 1:
        Y_train_1d = Y_train[:, 0]
    elif np.shape(Y_train)[1] > 1:
        Y_train_1d = Y_train.argmax(1)
    else:
        Y_train_1d = Y_train
    train_mask = np.isin(Y_train_1d, classes)

    if negate:
        train_mask = ~train_mask

    Y_train_1d = Y_train_1d[train_mask]
    X_train, Y_train = X_train[train_mask], Y_train[train_mask]

    X_new = []
    Y_new = []
    avail_classes, avail_idxs = np.unique(Y_train_1d, return_inverse=True)
    print("Adding samples for classes", avail_classes)
    for idx, cls in enumerate(avail_classes):
        cls_mask = avail_idxs == idx
        X_new.extend(X_train[cls_mask][:num_per_class])
        Y_new.extend(Y_train[cls_mask][:num_per_class])

    return np.concatenate([X_new], 0), np.concatenate([Y_new], 0)

