"""
@author: Md Rashad Al Hasan Rony

"""

from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from data_preprocessing import load_data,process_data,split_data

# Parameters
MAX_DOC_LENGTH = 25
BATCH_SIZE = 256
EPOCHS = 5


# Loading train and test dataset
x_train, y_train = load_data("train_data.csv", sample_ratio=0.1)
x_test, y_test = load_data("test_data.csv", sample_ratio=0.1)

# Data preprocessing
x_train, x_test, _ , n_vocab = process_data(x_train, x_test, MAX_DOC_LENGTH)


# Splitting dataset
x_test, x_val, y_test, y_val, _, test_size = split_data(x_test, y_test, 0.1)

# Model for training
model = Sequential()
model.add(Embedding((n_vocab+1), 15, input_length=MAX_DOC_LENGTH))
model.add(Bidirectional(LSTM(15)))
model.add(Dropout(0.5))
model.add(Dense(15, activation='sigmoid'))
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])


print('-------------------------------------Training Data------------------------------------\n')

model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=[x_val, y_val])
model.save('model_.h5')

# Loading the saved model
model = load_model('model_.h5')
pred = model.predict(x_test)

print('\n------------------------------------Evaluationg-----------------------------------------\n')
test_loss, accuracy = model.evaluate(x_test, y_test)
print('Test loss is: ',(test_loss*100),'%')
print('Accuracy is: ',(accuracy*100),'%')

