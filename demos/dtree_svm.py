from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
print(iris.data[107], iris.target_names[iris.target[107]])

my_svm  = SVC(kernel='poly', degree=2)
my_svm  = my_svm.fit(iris.data, iris.target)

my_tree = DecisionTreeClassifier()
my_tree = my_tree.fit(iris.data, iris.target)

pred = my_svm.predict([iris.data[107]])
print(iris.data[107], iris.target_names[iris.target[107]], iris.target_names[pred])

# Loop through SVM
print('SVM')
print('i  | data       | label     |  prediction')
svm_correct = 0
for i in range(len(iris.data)):
    pred = my_svm.predict([iris.data[i]])
    print(i, iris.data[i], iris.target_names[iris.target[i]], iris.target_names[pred])

    if pred[0] == iris.target[i]:
        svm_correct += 1

# Loop through tree
print('Decision tree')
print('i   data        label       prediction')
tree_correct = 0
for i in range(len(iris.data)):
    pred = my_tree.predict([iris.data[i]])
    print(i, iris.data[i], iris.target_names[iris.target[i]], iris.target_names[pred])

    if pred[0] == iris.target[i]:
        tree_correct += 1

print(f'SVM Accuracy = {svm_correct/len(iris.data) * 100}%')
print(f'Tree Accuracy= {tree_correct/len(iris.data) * 100}%')


'''
[[122232, 1, 5, 6, 8, 9, 9, 6],
 [123214, 5, 5, 6 , 7, 8, 9, 4],]


label = data[:][-1]
params = data[:][1:-2]
'''
