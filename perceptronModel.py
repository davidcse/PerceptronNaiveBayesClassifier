import copy
import random
from functools import reduce
debug = 1

class Perceptron:
    def __init__(self, learning_rate, converge_threshold):
        self.learning_rate = learning_rate
        self.converge_threshold = converge_threshold
        self.weights_dict = {}
        self.hard_stop = False

    def activation_function(self,linear_sum):
        if(linear_sum > 0):
            return 1
        else:
            return -1

    def classify(self,feature_dict):
        linear_sum = 0
        for key in feature_dict:
            if key not in self.weights_dict:
                self.weights_dict[key] = random.uniform(-1,1)
            linear_sum +=  self.weights_dict[key] * feature_dict[key]
        return self.activation_function(linear_sum)

    def train(self,feature_dict, expected_result):
        # test if reached convergence, cease training.
        if(self.hard_stop):
            return
        linear_sum = 0
        for key in feature_dict:
            # first time encounter this word. Same as if its weight's never been updated since initialization.
            if key not in self.weights_dict:
                self.weights_dict[key] = random.uniform(-1,1)
            #update the activation threshold
            linear_sum += self.weights_dict[key] * feature_dict[key]
        result =  self.activation_function(linear_sum)
        error = expected_result - result
        # Update the weights based on the errors, if an error occurred
        if(error):
            delta_weight_arr = [0] * len(feature_dict)
            for key in feature_dict:
                delta_weight = self.learning_rate * error * feature_dict[key]
                delta_weight_arr.append(delta_weight)
                # print("error:"+str(error) + "\tinput : "+ str(feature_dict[key]) +"\tdelta_weight : " + str(delta_weight))
                self.weights_dict[key] += delta_weight
            delta_weight_sum = reduce(lambda x,y: abs(x)+abs(y),delta_weight_arr)
            if(len(delta_weight_arr) > 0):
                delta_weight_avg = delta_weight_sum / len(delta_weight_arr)
                if(abs(delta_weight_avg) < self.converge_threshold):
                    print("\n\nCONVERGED delta_weight_avg is "+ str(delta_weight_avg) + "\n\n")
                    self.hard_stop = True

    def reset(self):
        for key in self.weights_dict:
            self.weights_dict[key] = random.uniform(-1,1)
        self.hard_stop = False
