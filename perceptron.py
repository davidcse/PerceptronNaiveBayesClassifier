import copy
import random

class Perceptron:
    def __init__(self, zeroed_vector_dict, learning_rate):
        self.zeroed_vector_dict = zeroed_vector_dict
        self.learning_rate = learning_rate
        self.weights_dict = copy.deepcopy(self.zeroed_vector_dict)
        self.reset()

    def activation_function(self,linear_sum):
        if(linear_sum > 0):
            return 1
        else:
            return -1

    def classify(self,file_path):
        fHandle = open(file_path,'r')
        text = fHandle.read().split()
        fHandle.close()
        file_word_vector = copy.deepcopy(self.zeroed_vector_dict)
        for w in text:
            file_word_vector[w] = file_word_vector[w] + 1
        linear_sum = 0
        for key in self.weights_dict:
            input_sum = self.weights_dict[key] * file_word_vector[key]
            linear_sum += input_sum
        return self.activation_function(linear_sum)

    def train(self,file_path, expected_result):
        fHandle = open(file_path,'r')
        text = fHandle.read().split()
        fHandle.close()
        file_word_vector = copy.deepcopy(self.zeroed_vector_dict)
        for w in text:
            file_word_vector[w] = file_word_vector[w] + 1
        linear_sum = 0
        for key in self.weights_dict:
            linear_sum += self.weights_dict[key] * file_word_vector[key]
        result =  self.activation_function(linear_sum)
        error = expected_result - result
        for input_key in file_word_vector:
            delta_weight = self.learning_rate * error * file_word_vector[input_key]
            self.weights_dict[input_key] += delta_weight

    def reset(self):
        for key in self.weights_dict:
            self.weights_dict[key] = random.uniform(-1,1)
