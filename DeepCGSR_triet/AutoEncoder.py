import keras
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class Autoencoder:
    def __init__(self, latent_dim=32, activation='relu', epochs=200, batch_size=120):
        self.latent_dim = latent_dim
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.his = None

    def _compile(self, input_dim):
        input_vec = Input(shape=(input_dim,))
        encoded = Dense(self.latent_dim, activation=self.activation)(input_vec)
        decoded = Dense(input_dim, activation=self.activation)(encoded)

        self.autoencoder = Model(input_vec, decoded)
        self.encoder = Model(input_vec, encoded)

        encoded_input = Input(shape=(self.latent_dim,))
        self.decoder = Model(encoded_input, self.autoencoder.layers[-1](encoded_input))

        self.autoencoder.compile(optimizer='adam', loss=keras.losses.mean_absolute_error)

    def fit(self, X):
        if not self.autoencoder:
            self._compile(X.shape[1])
        X_train, X_test = train_test_split(X, test_size=0.2)  # Adjust test_size as needed
        self.his = self.autoencoder.fit(
            X_train, X_train,
            epochs=self.epochs, batch_size=self.batch_size,
            shuffle=True, validation_data=(X_test, X_test)
        )