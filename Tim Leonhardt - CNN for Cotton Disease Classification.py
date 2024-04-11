import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from keras_tuner.tuners import RandomSearch
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Dense 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_fscore_support



datagenerator = ImageDataGenerator(rescale = 1.0/255,
                                        zoom_range = 0.5,
                                        horizontal_flip = False,
                                        width_shift_range=0.2,
                                        brightness_range=[0.1,1.5]
                                        )

train_data = datagenerator.flow_from_directory('../input/cotton-disease-dataset/Cotton Disease/train',
                                                    target_size = (256,256),
                                                    batch_size = 30,
                                                    class_mode = 'categorical')
val_data = datagenerator.flow_from_directory('../input/cotton-disease-dataset/Cotton Disease/val',
                                                    target_size = (256,256),
                                                    batch_size = 30,
                                                    class_mode = 'categorical')
test_data = datagenerator.flow_from_directory('../input/cotton-disease-dataset/Cotton Disease/test',
                                                    target_size = (256,256),
                                                    batch_size = 30,
                                                    class_mode = 'categorical')



model = tf.keras.models.Sequential([
    layers.Conv2D(filters=64,padding = "same",kernel_size=3,activation='relu',input_shape=[256,256,3]),
    layers.Conv2D(filters=32,padding = "same",kernel_size=3,activation='relu'),
    layers.MaxPool2D(pool_size=2,strides=2),
    layers.Conv2D(filters=32,padding = "same",kernel_size=3,activation='relu'),
    layers.Conv2D(filters=16,padding = "same",kernel_size=3,activation='relu'),
    layers.MaxPool2D(pool_size=2,strides=2),
    layers.Conv2D(filters=8,padding = "same",kernel_size=3,activation='relu'),
    layers.Conv2D(filters=8,padding = "same",kernel_size=3,activation='relu'),
    layers.MaxPool2D(pool_size=2,strides=2),
    layers.Flatten(),
    layers.Dense(units=128,activation='relu'),
    layers.Dense(units=128,activation='relu'),
    layers.Dense(units=4,activation='softmax'),])



model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
history = model.fit(x = train_data, validation_data = val_data, epochs = 100, callbacks=[early_stopping])



history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot()
history_frame.loc[:, ['accuracy', 'val_accuracy']].plot();



# prediction using test data 

y_pred_prob = model.predict(test_data)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = test_data.classes
class_names = ['Diseased cotton leaf', 'Diseased cotton plant', 'Fresh cotton leaf', 'Fresh cotton plant']

conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
plt.figure(figsize=(8, 6))

precision = precision_score(y_true, y_pred, average='weighted')
print(f'Precision: {precision:.4f}')

n_classes = 4
y_test_bin = label_binarize(y_true, classes=[0, 1, 2, 3])

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
colors = ['blue', 'red', 'green', 'purple']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve of class {0} (area = {1:0.2f})'.format(class_names[i], roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

roc_auc_macro = auc(fpr[2], tpr[2])
print("Area under ROC (Macro):", roc_auc_macro)

precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1_score:.4f}')



# Based on terrible results, decided to use train and validation data to test the predicting ability of the model
# to check for under- and overfitting. 



#prediction using training data 

y_pred_prob = model.predict(train_data)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = train_data.classes
class_names = ['Diseased cotton leaf', 'Diseased cotton plant', 'Fresh cotton leaf', 'Fresh cotton plant']

conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
plt.figure(figsize=(8, 6))

precision = precision_score(y_true, y_pred, average='weighted')
print(f'Precision: {precision:.4f}')

n_classes = 4
y_test_bin = label_binarize(y_true, classes=[0, 1, 2, 3])

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
colors = ['blue', 'red', 'green', 'purple']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve of class {0} (area = {1:0.2f})'.format(class_names[i], roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

roc_auc_macro = auc(fpr[2], tpr[2]) 
print("Area under ROC (Macro):", roc_auc_macro)

precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1_score:.4f}')



#prediction using validation data

y_pred_prob = model.predict(val_data)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = val_data.classes
class_names = ['Diseased cotton leaf', 'Diseased cotton plant', 'Fresh cotton leaf', 'Fresh cotton plant']

conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
plt.figure(figsize=(8, 6))

precision = precision_score(y_true, y_pred, average='weighted')
print(f'Precision: {precision:.4f}')

n_classes = 4
y_test_bin = label_binarize(y_true, classes=[0, 1, 2, 3])

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
colors = ['blue', 'red', 'green', 'purple']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve of class {0} (area = {1:0.2f})'.format(class_names[i], roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

roc_auc_macro = auc(fpr[2], tpr[2])  
print("Area under ROC (Macro):", roc_auc_macro)

precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1_score:.4f}')