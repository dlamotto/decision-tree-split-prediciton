# Danielle LaMotto

import numpy as np
import math


def counter(y):  # counts the amount of zeros and ones
    """Counts the classifiers amount of zeroes and ones

        Args:
            y: a list of classifiers either being 0 or 1

        Returns:
            The amount of zeros and the amount of ones
        """
    total_zeroes = 0
    total_ones = 0

    for element in y:
        if element == 0.0:
            total_zeroes = total_zeroes + 1
        else:
            total_ones = total_ones + 1

    return total_zeroes, total_ones


def entropy(classifiers):
    """Calculates the entropy of a set of classifiers

            Args:
                classifiers: a list of classifiers either being 0 or 1

            Returns:
                The entropy of a set of classifiers
            """
    total_zeroes, total_ones = counter(classifiers)
    total = total_zeroes + total_ones
    prob_of_one = total_ones / total  # P(1|dataset)
    prob_of_zero = total_zeroes / total  # P(0|dataset)

    if prob_of_one == 0:
        log2one = 0
    else:
        log2one = math.log2(prob_of_one)  # log base 2 of P(1|dataset)

    if prob_of_zero == 0:
        log2zero = 0
    else:
        log2zero = math.log2(prob_of_zero)  # log base 2 of P(0|dataset)

    answer = -((prob_of_one*log2one) + (prob_of_zero*log2zero))  # Entropy formula
    return answer


def find_column(x, i):
    """Gathers all attributes from a column into a list

            Args:
                x: a list of all attributes
                i: the index of the column to find

            Returns:
                A list of all the attributes found under the column index i
            """
    column = []  # holds column values of attributes[index_attribute]
    for lists in x:  # appends the value found in the attribute list at index_attribute
        column.append(lists[i])
    return column


def find_split(dataset, index_attribute, split_value):
    """Separates the classifiers into two lists. One list being all classifier values <= to the split value and the
    other being classifier values > the split value.

            Args:
                dataset: a dataset, tuple (X, y) where X is the data, y the classes
                index_attribute: the index of the attribute (column of X) to split on
                split_value: value of the attribute at index_attribute to split at
            Returns:
                A list of classifiers <= split_value and a list of classifiers > split_value
            """
    a = dataset[0]
    c = dataset[1]
    column = find_column(a, index_attribute)
    greater_indexes = []
    lower_indexes = []

    less_than_split = [i for i, a in enumerate(column) if a <= split_value]  # index vals of attributes <= split_value
    greater_than_split = [i for i, a in enumerate(column) if a > split_value]  # index vals of attributes > split_value

    for j in less_than_split:
        lower_indexes.append(c[j])  # classifier values for attributes <= split_value
    for k in greater_than_split:
        greater_indexes.append(c[k])  # classifier values for attributes > split_value

    return lower_indexes, greater_indexes


def avg_entropy(dataset, index_attribute, split_value):  # load(train), 8, 5
    """Calculates the average entropy as a split measure

                Args:
                    dataset: a dataset, tuple (X, y) where X is the data, y the classes
                    index_attribute: the index of the attribute (column of X) to split on
                    split_value: value of the attribute at index_attribute to split at
                Returns:
                    The average entropy of the resulting partition
                """
    # classifier values for attributes <= split_value & for attributes > split_value
    lower_indexes, greater_indexes = find_split(dataset, index_attribute, split_value)

    lower_entropy = entropy(lower_indexes)  # entropy of attributes <= split_value
    greater_entropy = entropy(greater_indexes)  # entropy of attributes > split_value

    total = len(greater_indexes) + len(lower_indexes)  # total number of all attributes
    zeroes_l, ones_l = counter(lower_indexes)  # number of zeroes and ones <= split_value
    zeroes_g, ones_g = counter(greater_indexes)  # number of zeroes and ones > split_value

    answer = (((zeroes_l+ones_l)/total)*lower_entropy) + (((zeroes_g+ones_g)/total)*greater_entropy)  # average entropy
    return answer


def IG(D, index, value):
    """Compute the Information Gain of a split on attribute index at value
    for dataset D.

    Args:
        D: a dataset, tuple (X, y) where X is the data, y the classes
        index: the index of the attribute (column of X) to split on
        value: value of the attribute at index to split at

    Returns:
        The value of the Information Gain for the given split
    """

    total_entropy = entropy(D[1])  # returns the entropy of the dataset
    average_entropy = avg_entropy(D, index, value)  # returns the average entropy based on a split value
    gain = total_entropy - average_entropy  # information gain formula

    return gain


def gini(classifiers):
    """Calculates the Gini Index formula

                Args:
                     classifiers: a list of classifiers either being 0 or 1
                Returns:
                    The gini index
                """
    total_zeroes, total_ones = counter(classifiers)
    total = total_zeroes + total_ones
    prob_of_one = total_ones / total  # P(1|dataset)
    prob_of_zero = total_zeroes / total  # P(0|dataset)

    answer = 1 - (pow(prob_of_one, 2) + pow(prob_of_zero, 2))	  # G(D), formula
    return answer


def G(D, index, value):
    """Compute the Gini index of a split on attribute index at value
    for dataset D.

    Args:
        D: a dataset, tuple (X, y) where X is the data, y the classes
        index: the index of the attribute (column of X) to split on
        value: value of the attribute at index to split at

    Returns:
        The value of the Gini index for the given split
    """

    # classifier values for attributes <= split_value & for attributes > split_value
    lower_indexes, greater_indexes = find_split(D, index, value)

    lower_gini = gini(lower_indexes)  # Gini of attributes <= split_value
    greater_gini = gini(greater_indexes)  # Gini of attributes > split_value
    total = len(greater_indexes) + len(lower_indexes)  # total number of all attributes

    zeroes_l, ones_l = counter(lower_indexes)  # number of zeroes and ones <= split_value
    zeroes_g, ones_g = counter(greater_indexes)  # number of zeroes and ones > split_value

    # weighted gini
    answer = (((zeroes_l + ones_l) / total) * lower_gini) + (((zeroes_g + ones_g) / total) * greater_gini)
    return answer


def CART(D, index, value):
    """Compute the CART measure of a split on attribute index at value
    for dataset D.

    Args:
        D: a dataset, tuple (X, y) where X is the data, y the classes
        index: the index of the attribute (column of X) to split on
        value: value of the attribute at index to split at

    Returns:
        The value of the CART measure for the given split
    """

    # classifier values for attributes <= split_value & for attributes > split_value
    lower_indexes, greater_indexes = find_split(D, index, value)
    zeroes_l, ones_l = counter(lower_indexes)
    zeroes_g, ones_g = counter(greater_indexes)

    prob_of_one_l = ones_l / (zeroes_l + ones_l)  # P(1|dataset_lower)
    prob_of_one_g = ones_g / (zeroes_g + ones_g)  # P(1|dataset_greater)

    prob_of_zero_l = zeroes_l / (zeroes_l + ones_l)  # P(0|dataset_lower)
    prob_of_zero_g = zeroes_g / (zeroes_g + ones_g)  # P(0|dataset_greater)

    total = len(greater_indexes) + len(lower_indexes)  # total number of all attributes

    first_half_formula = 2 * ((zeroes_l + ones_l) / total) * ((zeroes_g + ones_g) / total)
    second_half_ones = abs(prob_of_one_l - prob_of_one_g)
    second_half_zeros = abs(prob_of_zero_l - prob_of_zero_g)
    answer = first_half_formula * (second_half_ones + second_half_zeros)   # cart formula
    return answer


def partition(array, start, end):
    """Takes the last element as the pivot point, sorts it into its correct spot into the array and then sorts those
     values smaller to the pivot to the left and those larger to the right

       Args:
           array: A list of unsorted values
           start: Starting index, usually 0
           end: Ending index, usually the length of the array - 1

       Returns:
           The index where the partition was done
       """
    pivot = array[end]
    i = start-1  # pointer to the larger element

    for j in range(start, end):   # traverses through all elements of array
        if array[j] <= pivot:  # compares each element to the pivot
            i = i+1
            (array[i], array[j]) = (array[j], array[i])  # swapping element i with j
    (array[i+1], array[end]) = (array[end], array[i+1])  # swapping the pivot with i

    # partition position
    return i+1


def quickSort(array, start, end):
    """Sorts an array by ascending order

       Args:
           array: A list of unsorted values
           start: Starting index, usually 0
           end: Ending index, usually the length of the array - 1

       Returns:
           The sorted array
       """
    if start < end:
        point = partition(array, start, end)
        quickSort(array, start, point-1)  # recursive call for left of pivot
        quickSort(array, point+1, end)  # recursive call for right of pivot
    return array


def bestSplit(D, criterion):
    """Computes the best split for dataset D using the specified criterion

    Args:
        D: A dataset, tuple (X, y) where X is the data, y the classes
        criterion: one of "IG", "GINI", "CART"

    Returns:
        A tuple (i, value) where i is the index of the attribute to split at value
    """
    result = []
    column = []  # holds column values of attributes[index_attribute]
    for lists in D[0]:
        for i in range(0, len(lists)):
            column.append(find_column(D[0], i))  # sorts the attributes into a list via column
    largest = None	 # comparison variable, to find the largest value
    smallest = None	 # comparison variable, to find the smallest value
    for j in column:	  # loops through attributes from the column-based list
        sorted_list = quickSort(j, 0, len(j) - 1)  # returns the list in ascending order
        max_val = sorted_list[len(sorted_list) - 1]
        for k in j:		# loops through the actual attributes in the list
            if k != max_val:	  # ignores the max attribute
                if criterion == "IG":
                    option = IG(D, column.index(j), k)   # value of IG
                    if largest is None or option > largest:	   # compares all the options until it finds the largest
                        largest = option
                        store_index = column.index(j)
                        store_value = k
                elif criterion == "GINI":
                    option = G(D, column.index(j), k)   # value of GINI
                    result.append(option)
                    if smallest is None or option < smallest:  # compares all the options until it finds the smallest
                        smallest = option
                        store_index = column.index(j)
                        store_value = k
                elif criterion == "CART":
                    option = CART(D, column.index(j), k)    # value of CART
                    if largest is None or option > largest:   # compares all the options until it finds the largest
                        largest = option
                        store_index = column.index(j)
                        store_value = k
                else:
                    print("Not a criterion")
    return store_index, store_value			# returns index and value based on the criterion's best split


def load(filename):
    """Loads filename as a dataset. Assumes the last column is classes, and
    observations are organized as rows.

    Args:
        filename: file to read

    Returns:
        A tuple D=(X,y), where X is a list or numpy ndarray of observation attributes
        where X[i] comes from the i-th row in filename; y is a list or ndarray of
        the classes of the observations, in the same order
    """
    filename = open(filename, 'r')  # opens file
    attributes = []  # holds attributes
    classifier = []  # hold classifiers
    chunked_list = []  # holder for when attributes and classifiers get separated
    chunk_size = 10

    for lines in filename:  # takes the first 10 items in a line as a list
        y = lines.split(",")
        for i in range(0, len(y), chunk_size):
            chunked_list.append(y[i:i + chunk_size])

    for element in chunked_list:  # based on the index, add classifiers to one list and attributes to another
        if chunked_list.index(element) % 2:
            classifier.append(element)
        else:
            attributes.append(element)

    new = list(np.concatenate(classifier))  # flattens lists of classifiers

    for i in range(0, len(new)):  # changes all values in the list from a string to a float
        new[i] = float(new[i])
    for lists in attributes:
        for i in range(0, len(lists)):  # changes all values in the list from a string to a float
            lists[i] = float(lists[i])

    data_tuple = (attributes, new)
    return data_tuple  # returns a list of lists of attributes and a list of classifiers


def classifyIG(train, test):
    """Builds a single-split decision tree using the Information Gain criterion
    and dataset train, and returns a list of predicted classes for dataset test

    Args:
        train: a tuple (X, y), where X is the data, y the classes
        test: the test set, same format as train

    Returns:
        A list of predicted classes for observations in test (in order)
    """

    index, split = bestSplit(train, "IG")  # test: 2, 0, train:7, 4
    classes = []

    column = find_column(test[0], index)  # returns the column of attributes at training sets best split index
    for val in column:  	# creates a new list based on the best split from the training data
        if val <= split:
            classes.append(0)
        else:
            classes.append(1)

    print(f"The best split occurs at the value: {split} found at the index: {index}")
    print("Classifiers for test data based on training sets best split (IG):", classes)
    print("\n")


def classifyG(train, test):
    """Builds a single-split decision tree using the GINI criterion
    and dataset train, and returns a list of predicted classes for dataset test

    Args:
        train: a tuple (X, y), where X is the data, y the classes
        test: the test set, same format as train

    Returns:
        A list of predicted classes for observations in test (in order)
    """

    index, split = bestSplit(train, "GINI")  # test: 2, 0, train:5, 4
    classes = []

    column = find_column(test[0], index)  # returns the column of attributes at training sets best split index
    for val in column:  	# creates a new list based on the best split from the training data
        if val <= split:
            classes.append(0)
        else:
            classes.append(1)

    print(f"The best split occurs at the value: {split} found at the index: {index}")
    print("Classifiers for test data based on training sets best split (GINI):", classes)
    print("\n")


def classifyCART(train, test):
    """Builds a single-split decision tree using the CART criterion
    and dataset train, and returns a list of predicted classes for dataset test

    Args:
        train: a tuple (X, y), where X is the data, y the classes
        test: the test set, same format as train

    Returns:
        A list of predicted classes for observations in test (in order)
    """

    index, split = bestSplit(train, "CART")  # test: 2, 0, train:2,0
    classes = []

    column = find_column(test[0], index)  # returns the column of attributes at training sets best split index
    for val in column:  	# creates a new list based on the best split from the training data
        if val <= split:
            classes.append(1)
        else:
            classes.append(0)

    print(f"The best split occurs at the value: {split} found at the index: {index}")
    print("Classifiers for test data based on training sets best split (CART): ", classes)
    print("\n")


def main():
    D = load('train.txt')  # training set tuple where x is the data and y is the classifier
    Q = load('test2.txt')  # test set tuple where z is the data and w is the classifier
    classifyIG(D, Q)
    classifyG(D, Q)
    classifyCART(D, Q)


if __name__=="__main__":
    main()
