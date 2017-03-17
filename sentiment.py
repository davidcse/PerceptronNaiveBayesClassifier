from sys import argv
import preprocessor as pp
import perceptron
import copy
from sys import argv

largest_dict  = copy.deepcopy(pp.all_words)
for key in largest_dict:
    largest_dict[key] = 0

perceptron = perceptron.Perceptron(largest_dict,0.005)
combined_data = []
pp.interleave_pos_neg_files(pp.neg_files_basepath,pp.negative_files,pp.pos_files_basepath,pp.positive_files, combined_data, 1)

################################################
#       TEST
################################################
debug = 0
verbose = 0
for i in argv:
    if i == "-d":
        debug = 1
    if i == "-v":
        verbose = 1

if(debug):
    combined_data = combined_data[:len(combined_data)//4]

accuracy_report = []
for i in range(10):
    for i in range(5):
        fifth_partition = len(combined_data)//5
        start_partition = i*fifth_partition
        end_partition = (i+1) * fifth_partition
        print("\n\nTRAINING:")
        print("______________________________________________________________________________________________")
        print("start_partition:" + str(start_partition))
        print("end_partition:" + str(end_partition))
        test_set = combined_data[start_partition:end_partition]
        train_set = combined_data[:start_partition] + combined_data[end_partition:]

        finished = 0
        for train_tuple in train_set:
            training_file = train_tuple[0]
            expected_result = train_tuple[1]
            perceptron.train(training_file,expected_result)
            finished = finished + 1
            if verbose: print("perceptron finished training on \t" + training_file + "\tremaining files:" + str(len(train_set)- finished))

        correct = []
        incorrect = []
        tested = 0
        for test_tup in test_set:
            result = perceptron.classify(test_tup[0])
            tested = tested + 1
            if result == test_tup[1]:
                correct.append(result)
                output = "CORRECT"
            else:
                incorrect.append(result)
                output = "WRONG"
            if verbose: print("perceptron finished testing on \t" + test_tup[0] + "\tremaining files:" + str(len(test_set)- tested) + "\tRESULT:"+ output)


        if verbose: print("CORRECT PERCENT: " + str(len(correct)/(len(correct) + len(incorrect))))
        if verbose: print("INCORRECT PERCENT: " + str(len(incorrect)/(len(correct) + len(incorrect))))
        accuracy_report.append(len(correct)/(len(correct) + len(incorrect)))
        perceptron.reset()

print("ACCURACY REPORT:")
for i in range(len(accuracy_report)):
    print("TEST # "+str(i+1) + " : " + str(accuracy_report[i]) + "%")

total = 0
for i in accuracy_report:
    total += i

print("AVERAGE OVER ALL TRIALS: " + str(total/len(accuracy_report)))
