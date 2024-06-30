from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np
import matplotlib.pyplot as plt

def decoder_review(reverse_word_index, data, ind):
    decoded_review = ' '.join([reverse_word_index[i] for i in data[ind]])
    return decoded_review

def vectorize(data, dimension=10000):
    result = np.zeros((len(data), dimension), dtype=int)
    for i,j in enumerate(data):
        result[i,j]=1
    return result

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
print(x_train[0])
print(y_train[0])

word_index = imdb.get_word_index()
print(word_index)
reverse_word_index = dict([(value,key) for (key, value) in word_index.items()])
for i in range(1,21):
    print(i,'=', reverse_word_index[i])

print(decoder_review(reverse_word_index, x_train, 0))
x_train = vectorize(x_train)
x_test = vectorize(x_test)

print(x_train[0])
print(len(x_train[0]))
print(x_train.shape)

model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(10000,)))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

text_model = model.fit(x_train,
                       y_train,
                       epochs=20,
                       batch_size=128,
                       validation_split=0.1)

plt.plot(text_model.history['accuracy'],
         label='Обучающий набор')
plt.plot(text_model.history['val_accuracy'],
         label='Проверочный набор')
plt.xlabel('Эпохи')
plt.ylabel('Точность')
plt.legend()
plt.show()

final_score = model.evaluate(x_test, y_test)
print(final_score)
