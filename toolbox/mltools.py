from tensorflow.keras import models,layers

def init_nn_binary_classifier():

    ### Model architecture
    model = models.Sequential()
    model.add(layers.Dense(100, input_dim=2, activation='relu'))
    ### size 1 (predict one value):
    model.add(layers.Dense(1, activation='sigmoid'))

    ### Model optimization : Optimizer, loss and metric
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
