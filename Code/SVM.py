from sklearn import metrics
from sklearn import svm
import matplotlib.image as mpimg
import numpy as np

# train data preprocessing
f = open('dataKaggle/train.txt')
train_labels = []
train_images = []
cnt = 0
for i in f:
    inp = i.split(',')
    path = 'dataKaggle/train/' + inp[0]
    train_images.append(mpimg.imread(path).reshape(32*32))
    train_labels.append(int(inp[1]))
f.close()

# validation data preprocessing
f = open('dataKaggle/validation.txt')
validation_labels = []
validation_images = []
cnt = 0
for i in f:
    inp = i.split(',')
    path = 'dataKaggle/validation/' + inp[0]
    validation_images.append(mpimg.imread(path).reshape(32*32))
    validation_labels.append(int(inp[1]))
f.close()
validation_labels_exp = validation_labels

train_images = np.asarray(train_images)
validation_images = np.asarray(validation_images)

# processing

# 60% accuracy SVC model with linear kernel
# SVM = svm.SVC(C=1.0, kernel='linear')
# SVM.fit(train_images, train_labels)
# predictions = SVM.predict(validation_images)
# accuracy = metrics.accuracy_score(validation_labels, predictions)

# 73,5% accuracy SVC model with default parameters
SVM = svm.SVC()
SVM.fit(train_images, train_labels)
predictions = SVM.predict(validation_images)
accuracy = metrics.accuracy_score(validation_labels, predictions)

print(accuracy)


# confusiong matrix
def confusion_matrix(y_true, y_pred):
    conf_matrix = np.zeros((10, 10))
    for i in range(len(y_true)):
        conf_matrix[y_true[i], y_pred[i]] += 1
    print(conf_matrix)


confusion_matrix(validation_labels, predictions)
