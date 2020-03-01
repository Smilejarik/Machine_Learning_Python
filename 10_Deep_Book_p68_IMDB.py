from keras.datasets import imdb
from keras import models, layers
import numpy as np
import matplotlib.pyplot as plt


(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print("Train len: {}, data 0: {}".format(len(train_data), train_data[0]))
print("Test len: {}, data 0: {}".format(len(test_data), test_data[0]))
print(train_labels[0])


# decode review back to words:
def decode_review(rev_index):
    word_index = imdb.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])  # key is word, value is number
    decoded_review = ' '.join([reverse_word_index.get(i-3, '?') for i in train_data[rev_index]])
    print(decoded_review)


def vectorize_sequences(sequenses, dimension=10000):
    results = np.zeros((len(sequenses), dimension))
    print(results.shape)
    for i, sequense in enumerate(sequenses):
        #print("i: {}, sequense: {}".format(i, sequense))
        #print("len sequense: {}".format(len(sequense)))
        results[i, sequense] = 1
        #if i > 2:
            #break
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
print(x_train[0])
print(x_test[0])

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
print(y_train[0])

#  Here is model starts
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

#  Separate training and validation data: (WHY? we have test data already)
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

#  train our model
history = model.fit(x_train,
                    y_train,
                    epochs=4,
                    batch_size=512,
          validation_data=(x_test, y_test))

history_dict = history.history  # dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])
#print("Results: {}".format(history_dict['val_acc'][:-1]))
#print("Predicted 5: {}, 15: {}, 25: {}".format(model.predict(x_test[5]), model.predict(x_test[15]), model.predict(x_test[25])))

#  plotting results
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc = history_dict['acc']
val_acc_values = history_dict['val_acc']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training Loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# second plot
plt.clf()

plt.plot(epochs, acc, 'bo', label='Training Accuracy')
plt.plot(epochs, val_acc_values, 'b', label='Validation Accuracy')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
