import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


def gaussian_pdf(x, mean, deviation):
    """
    Calculates the PDF of a Gaussian distribution.
    :param x: The x value.
    :param mean: The mean.
    :param deviation: The standard deviation.
    :return: A probability [0, 1]
    """
    variance = np.square(deviation)
    coefficient = 1 / np.sqrt(2 * np.pi * variance)
    exp = np.exp(-(np.square((x - mean))) / (2 * variance))
    return coefficient * exp


class Model:
    """
    This class wraps a simple Bayesian Naive Algorithm (supervised learning).
    When it is created, the training and test data is imported, and the model is trained (parameters estimated).
    """
    def __init__(self, percentage):
        """
        First, this will generate the training and test data.
        Then, it will estimate the parameters based off of the training data.
        :param percentage: The percentage of the total dataset to use for training.
        """
        self.percentage = percentage
        # Get the data from Iris data set.
        iris = datasets.load_iris()
        self.targets = iris.target_names
        self.features = iris.feature_names
        self.X_train, self.X_test, self.Y_train, self.Y_test = \
            train_test_split(iris.data, iris.target, test_size=percentage)
        # Compute the number of classes and features (this should not change).
        number_classes = len(iris.target_names)
        number_features = len(iris.feature_names)
        if number_classes == 0:
            raise RuntimeError("No classes")
        if number_features == 0:
            raise RuntimeError("No features")
        self.number_classes = number_classes
        self.number_features = number_features
        # Define means, standard deviations and class priors for each class.
        self.means = np.zeros((number_classes, number_features))
        self.deviations = np.zeros((number_classes, number_features))
        self.priors = np.zeros(number_classes)
        for class_idx in range(number_classes):
            # Get all the flowers of this class.
            class_data = self.X_train[np.where(self.Y_train == class_idx)]
            for feature_idx in range(number_features):
                # For every feature of that class, generate a mean and standard deviation for this class.
                self.means[class_idx, feature_idx] = np.mean(class_data[:, feature_idx])
                self.deviations[class_idx, feature_idx] = np.std(class_data[:, feature_idx])
            # Estimate class priors by using p(x | C_i) = len(C_i) / N.
            self.priors[class_idx] = class_data.shape[0] / self.X_train.shape[0]

    def posterior(self, x, class_idx, feature_idx):
        """
        Uses Bayes' Rule to estimate the posterior probability of a data point belonging to a class, given the following:
        - Probability to obtain the datapoint knowing it belongs to the class class_idx.
        - Class prior (estimated in training).
        - Total probability (can be calculated for all classes).
        :param x: The feature scalar.
        :param class_idx: The index of the class to calculate the posterior probability for.
        :param feature_idx: The index of the feature to use.
        :return: The posterior probability of the given class.
        """
        # p(Ci).
        prior = self.priors[class_idx]
        # p(x | Ci).
        probability = gaussian_pdf(x, self.means[class_idx, feature_idx], self.deviations[class_idx, feature_idx])
        # sum from 0 to n as k: p(x | Ck) * p(Ck).
        denominator = 0
        for k in range(self.number_classes):
            class_prior = self.priors[k]
            class_probability = gaussian_pdf(x, self.means[k, feature_idx], self.deviations[k, feature_idx])
            denominator += (class_probability * class_prior)
        # Bayes' Rule: p(x | Ci) * p(Ci) / sum from 0 to n as k: p(x | Ck) * p(Ck).
        return (probability * prior) / denominator

    def classify(self, x, feature_idx):
        """
        Classifies the feature scalar x.
        :param x: The scalar.
        :param feature_idx: The index of the feature to use to classify.
        :return: The resulting class, and the confidence (probability value).
        """
        class_probabilities = np.zeros(self.number_classes)
        # Go through every class.
        for class_idx in range(self.number_classes):
            # Figure out the probability of it being that class.
            class_probabilities[class_idx] = self.posterior(x, class_idx, feature_idx)
        # Find the highest probability class.
        result = np.argmax(class_probabilities)
        return result, class_probabilities[result]

    def evaluate(self):
        """
        Evaluates the performance of the model.
        :return: A tuple of percentages for each feature.
        """
        total = self.X_test.shape[0]
        # Test individually for every feature.
        for feature_idx in range(self.number_features):
            correct = 0
            for data_idx in range(total):
                test_feature = self.X_test[data_idx, feature_idx]
                test_result = self.classify(test_feature, feature_idx)
                if test_result[0] == self.Y_test[data_idx]:
                    correct += 1
            yield correct / total


def number_input(converter, criteria, error_message):
    """
    Asks the user for input, and validates it. Asks until a proper answer can be returned.
    :param converter: The function to convert the input to the required data type.
    :param criteria: The criteria in the required data type that the input needs to fulfill.
    :param error_message: The error message to show when the conversion is not sucessful/criteria is not fulfilled.
    :return: The user input in the correct data type.
    """
    while True:
        string = input()
        try:
            value = converter(string)
            if criteria(value):
                break
            else:
                print(error_message)
        except ValueError:
            print(error_message)
        print("Please try again")
    return value


def lambda_integer(x): int(x)
def lambda_float(x): float(x)
def lambda_percentage(x): round(x * 100, ndigits=2)


def main():
    model = None
    while True:
        training_print = "N/A" if model is None else model.percentage
        print(f"\nWelcome to Iris (trained to: {lambda_percentage(training_print)}%)")
        print("- Please select an option")
        print("1. See all classes and features")
        print("2. Train the model and estimate parameters")
        print("3. Evaluate the model")
        print("4. Classify an unknown Iris flower")
        print("5. Exit")
        selection = number_input(lambda_integer, lambda x: 1 <= x <= 5, "Invalid integer choice [1, 5]")
        if selection == 1:
            if model is None:
                print("Not trained yet, classes and features are unknown")
                continue
            print("Iris classes:")
            for index, class_name in enumerate(model.targets):
                print(f"{index}. {class_name}")
            print("Iris features:")
            for index, feature_name in enumerate(model.features):
                print(f"{index}. {feature_name}")
        elif selection == 2:
            print("Which percentage of the data should be used for training?")
            percentage = number_input(lambda_float, lambda x: 0.0 < x < 1.0, "Invalid percentage (0.0-1.0)")
            model = Model(percentage)
            print(f"Successfully trained with {percentage * 100}% of the data")
        elif selection == 3:
            if model is None:
                print("Not trained yet, cannot evaluate")
                continue
            print("Accuracy for respective features:")
            for index, result in enumerate(model.evaluate()):
                print(f"{index}. ({model.features[index]}) {lambda_percentage(result)}%")
        elif selection == 4:
            print(f"Enter feature [0-{model.number_features})")
            error_message = f"Invalid feature [0, {model.number_features})"
            feature_idx = number_input(lambda_integer, lambda x: 0 <= x < model.number_features, error_message)
            print("Enter the feature value")
            test = number_input(lambda_float, lambda x: True, "")
            result = model.classify(test, feature_idx)
            print(
                f"The Iris is a {model.targets[result[0]]}, with {lambda_percentage(result[1])}% probability")
        elif selection == 5:
            exit(0)


if __name__ == "__main__":
    main()
