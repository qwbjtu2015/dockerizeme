import tensorflow as tf
import numpy as np

data = np.array([
  [-2.0, -1],  # Alice
  [25, 6],   # Bob
  [17, 4],   # Charlie
  [-15, -6], # Diana
])
all_y_trues = np.array([
  1, # Alice
  0, # Bob
  0, # Charlie
  1, # Diana
])

inputs = tf.keras.Input(shape=(2,))
x = tf.keras.layers.Dense(2, use_bias=True)(inputs)
outputs = tf.keras.layers.Dense(1, use_bias=True, activation='sigmoid')(x)
m = tf.keras.Model(inputs, outputs)

m.compile(tf.keras.optimizers.SGD(learning_rate=0.1), 'mse')
m.fit(data, all_y_trues, epochs=1000, batch_size=1, verbose=0)

emily = np.array([[-7, -3]])
frank = np.array([[20, 2]])
print(m.predict(emily))
print(m.predict(frank))