from tensorflow.keras import models, optimizers, losses, activations
from tensorflow.keras.layers import *
import tensorflow as tf
import time


class Classifier(object):

    def __init__(self, config, number_of_classes, maxlen, X):
        dropout_rate = 0.5
        input_shape = (maxlen,)
        target_shape = (maxlen, 1)
        X = X
        MAX_NB_WORDS=50000
        EMBEDDING_DIM=100

        self.model_scheme = []

        if config.architecture == "LSTM":
            self.model_scheme = [
                Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1], trainable=False),
                SpatialDropout1D(config.dropout),
                LSTM(128, dropout=config.dropout, recurrent_dropout=0.2),
                LayerNormalization(axis=1),
                Dense(128, activation='relu'),
                Dropout(rate=0.2),
                Dense(number_of_classes, activation='softmax')
            ]
            self.__model = tf.keras.Sequential(self.model_scheme)
        elif config.architecture == "LSTM_bidirectional":
            # TODO: 32 still to much :(
            self.model_scheme = [
                    Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1], trainable=True),
                    SpatialDropout1D(config.dropout),
                    Bidirectional(LSTM(32, dropout=config.dropout, recurrent_dropout=0.2)),
                    LayerNormalization(axis=1),
                    # Dense(128, activation='relu'),
                    # Dropout(rate=0.2),
                    Dense(number_of_classes, activation='softmax')
                ]
            self.__model = tf.keras.Sequential(self.model_scheme)
        elif config.architecture == "LSTM_CNN":
            # TODO: make second sequence and try out without bidirectional, try IEMOCAP architecture
            sequence_input = Input(shape=input_shape, dtype='int32')

            embedding_layer_train = Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1], trainable=True)
            embedded_sequences_train = embedding_layer_train(sequence_input)
            drop1 = SpatialDropout1D(config.dropout)
            embedded_sequences_train_drop = drop1(embedded_sequences_train)
            l_lstm1 = Bidirectional(LSTM(6, return_sequences=True, dropout=0.3, recurrent_dropout=0.0))(embedded_sequences_train_drop)
            l_conv_2 = Conv1D(filters=24, kernel_size=2, activation='relu')(l_lstm1)
            l_conv_2 = Dropout(0.3)(l_conv_2)
            l_conv_3 = Conv1D(filters=24, kernel_size=3, activation='relu')(l_lstm1)
            l_conv_3 = Dropout(0.3)(l_conv_3)

            l_conv_5 = Conv1D(filters=24, kernel_size=5, activation='relu', )(l_lstm1)
            l_conv_5 = Dropout(0.3)(l_conv_5)
            l_conv_6 = Conv1D(filters=24, kernel_size=6, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(
                l_lstm1)
            l_conv_6 = Dropout(0.3)(l_conv_6)

            l_conv_8 = Conv1D(filters=24, kernel_size=8, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(
                l_lstm1)
            l_conv_8 = Dropout(0.3)(l_conv_8)

            conv_1 = [l_conv_6, l_conv_5, l_conv_8, l_conv_2, l_conv_3]

            l_lstm_c = Concatenate(axis=1)(conv_1)

            l_pool = MaxPooling1D(4)(l_lstm_c)
            l_drop = Dropout(0.5)(l_pool)
            l_flat = Flatten()(l_drop)
            l_dense = Dense(26, activation='relu')(l_flat)
            preds = Dense(number_of_classes, activation='softmax')(l_dense)
            self.__model = tf.keras.Model(sequence_input, preds)
        else:
            self.model_scheme = [
                Reshape(input_shape=input_shape, target_shape=target_shape),
                Conv1D(128, kernel_size=5, strides=1, activation=activations.relu),  # kernel_regularizer='l1'
                MaxPooling1D(pool_size=5),
                Conv1D(128, kernel_size=5, strides=1, activation=activations.relu),  # kernel_regularizer='l1'
                MaxPooling1D(pool_size=5),
                Conv1D(128, kernel_size=5, strides=1, activation=activations.relu),  # kernel_regularizer='l1'
                MaxPooling1D(pool_size=5),
                Flatten(),
                Dense(128, activation=activations.relu),
                BatchNormalization(),
                Dropout(dropout_rate),
                Dense(number_of_classes, activation=tf.nn.softmax)
            ]
            self.__model = tf.keras.Sequential(self.model_scheme)


        # compile model like you usually do.
        # notice use of config.
        optimizer = tf.keras.optimizers.Adam(config.learning_rate)
        self.__model.compile(
            optimizer=optimizer,
            loss=config.loss_function,
            metrics=['accuracy'],
        )
        print(self.__model.summary())

    def fit(self, X, Y, hyperparameters):
        initial_time = time.time()
        self.__model.fit(X, Y,
                         batch_size=hyperparameters['batch_size'],
                         epochs=hyperparameters['epochs'],
                         callbacks=hyperparameters['callbacks'],
                         validation_data=hyperparameters['val_data']
                         )
        final_time = time.time()
        eta = (final_time - initial_time)
        time_unit = 'seconds'
        if eta >= 60:
            eta = eta / 60
            time_unit = 'minutes'
        self.__model.summary()
        print('Elapsed time acquired for {} epoch(s) -> {} {}'.format(hyperparameters['epochs'], eta, time_unit))

    def evaluate(self, test_X, test_Y):
        return self.__model.evaluate(test_X, test_Y)

    def predict(self, X):
        predictions = self.__model.predict(X)
        return predictions

    def save_model(self, file_path):
        self.__model.save(file_path)

    def load_model(self, file_path):
        self.__model = models.load_model(file_path)
