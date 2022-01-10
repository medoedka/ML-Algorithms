import numpy as np
import random


class DataFormatError(Exception):
    pass


class DataTypeError(Exception):
    pass


class ArrayTypeError(Exception):
    pass


class EmptyValueError(Exception):
    pass


class InsufficientDataError(Exception):
    pass


class Kmeans():

    iteration = 0
    loss_dict = {}

    def __init__(self, data=None, cluster_kernels=None, k=3, cluster_convergence_step=1e-4):
        self.data = None
        self.cluster_kernels = None
        self.k = k
        self.cluster_convergence_step = cluster_convergence_step

    def __data_check(self):
        if not isinstance(self.data, np.ndarray):
            raise ArrayTypeError('Datatype must be numpy.ndarray')
        if len(self.data) < self.k:
            raise InsufficientDataError('Quantity of data points is less then number of clusters')
        for vector_index in range(1, len(self.data)):
            if len(self.data[vector_index]) != len(self.data[vector_index - 1]):
                raise DataFormatError('All data vectors must be the same length')
        for vector in self.data:
            for num in vector:
                if not isinstance(num, np.int64) and not isinstance(num, np.float64):
                    raise DataTypeError('All data entries must be numeric')
                if np.isnan(num):
                    raise EmptyValueError('Your data contains NaN values')

    def __initialization_of_clusters(self):
        cluster_kernels_coordinates = random.sample(range(len(self.data)), self.k)
        cluster_coordinates = np.array(list(map(lambda x: self.data[x], cluster_kernels_coordinates)))
        return cluster_coordinates

    def __find_minimum(self, kernels):
        minimums = []
        for vector in self.data:
            subresult = 0
            kernel_number = 0
            for kernel_index, kernel in enumerate(kernels):
                result = np.linalg.norm(vector - kernel) ** 2
                if subresult == 0:
                    subresult = result
                    kernel_number = kernel_index
                if result < subresult:
                    subresult = result
                    kernel_number = kernel_index
            minimums.append(kernel_number)
        return minimums

    def loss(self, loss_val):
        self.iteration += 1
        self.loss_dict[self.iteration] = loss_val

    def __count_loss(self, minimums, cluster_kernels):
        loss_val = 0
        for index, vector in enumerate(self.data):
            loss_val += np.linalg.norm(vector - cluster_kernels[minimums[index]]) / len(self.data)
        self.loss(loss_val)

    def __centroid_correction(self, minimums, cluster_coordinates):
        for data_pos_index, cluster_index in enumerate(set(minimums)):
            indices = [index for index, element in enumerate(minimums) if element == cluster_index]
            for position in range(len(self.data[0])):
                avg_data_value = 0
                counter = 0
                for data_index in indices:
                    avg_data_value += self.data[data_index][position]
                    counter += 1
                cluster_coordinates[cluster_index][position] = avg_data_value / counter

        return cluster_coordinates

    def fit(self, data):
        self.data = data
        self.__data_check()
        cluster_kernels = self.__initialization_of_clusters()
        while True:
            if len(self.loss_dict) < 2:
                pass
            else:
                if self.loss_dict[self.iteration - 1] - self.loss_dict[self.iteration] > self.cluster_convergence_step:
                    break
            minimums = self.__find_minimum(cluster_kernels)
            self.__count_loss(minimums, cluster_kernels)
            self.cluster_kernels = self.__centroid_correction(minimums, cluster_kernels)
        return cluster_kernels
        print(f'Number of iterations: {self.iteration}\n')
        print(f'Loss: {self.loss_dict[self.iteration]}\n')
        print(f'Cluster centers: {cluster_kernels}')
