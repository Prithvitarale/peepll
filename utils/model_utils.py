import tensorflow as tf
from utils.data_utils import datafilter, datafilter_perclass
from tensorflow import keras
import numpy as np
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input



# Function Description: Early Stopping Mechanism
# for Pre-training models. Not used in Lifelong Learning.
class TargetValAccuracyCallback(Callback):
    def __init__(self, min_val_accuracy, max_val_accuracy):
        super(TargetValAccuracyCallback, self).__init__()
        # self.min_val_accuracy = 0.7
        self.min_val_accuracy = 0.75
        self.max_val_accuracy = 0.85

    def on_epoch_end(self, epoch, logs=None):
        val_accuracy = logs.get('val_accuracy')
        train_accuracy = logs.get('accuracy')
        okay = False
        if train_accuracy is not None:
            if train_accuracy > 0.85:
                okay = True
                print("enough training")
                if val_accuracy is not None:
                    if val_accuracy >= self.max_val_accuracy and okay:
                        print(f"\nReached {self.max_val_accuracy*100}% val accuracy, stopping training ({train_accuracy}).")
                        self.model.stop_training = True
                    elif val_accuracy >= self.min_val_accuracy:
                        print(f"\nReached {self.min_val_accuracy*100}% val accuracy, considering stopping training ({train_accuracy}).")
                        # If within range, consider stopping if further improvement is minimal
                        self.model.stop_training = True

# Function Description: Samples a latent code, given z_mean and z_variance
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Function Description: Defines our VAE model without reconstruction.
class VAE(keras.Model):
    def __init__(self, input_shape, latent_dim=128, encoder=None, decoder=None, confidence_threshold=0.5, **kwargs):
        super(VAE, self).__init__(**kwargs)

        self.confidence_threshold = confidence_threshold

        if encoder is None:
            inputs = keras.Input(shape=(input_shape))
            vgg = VGG16(weights='imagenet', include_top=False)
            inputs = preprocess_input(inputs)
            x = vgg(inputs)
            x = tf.reshape(x, [-1, x.shape[1]*x.shape[2]*x.shape[3]])

            z_mean = keras.layers.Dense(latent_dim)(x)
            z_log_var = keras.layers.Dense(latent_dim)(x)
            z = Sampling()([z_mean, z_log_var])
            self.encoder = keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")
        else:
            self.encoder = encoder

        if decoder is None:
            inputs = keras.Input(shape=(latent_dim,))
            x = keras.layers.Dropout(0)(inputs)
            o = x
            o = keras.layers.Dense(256, activation='relu')(o)
            o = keras.layers.Dense(128)(o)
            o = keras.layers.Dense(100, activation="softmax", name = "classifier")(o)
            self.decoder = keras.Model(inputs, [o], name="decoder")
        else:
            self.decoder = decoder

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

        self.train_mode = 0
        self.memorized_data = []
        self.memorized_targets = []
        self.memorized_prediction = []

    @property
    def metrics(self):
        m = super().metrics
        return list(m) + [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, data, training=False):
        z_mean, z_log_var, z = self.encoder(data, training=training)
        classification = self.decoder(z, training=training)
        return classification, z, z_mean, z_log_var
    
    def train_step(self, data):
        data, confidences, targets = data
        # print('confd')

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data, training=True)
            reconstruction = self.decoder(z, training=True)
            a = keras.losses.categorical_crossentropy(targets, reconstruction)
            reconstruction_loss = tf.reduce_mean(a)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            if self.train_mode == 0:
                # total_loss = reconstruction_loss + tf.reduce_mean(0.00396*(kl_loss)) # minimgnet
                total_loss = reconstruction_loss + tf.reduce_mean(0.007*(kl_loss)) # cifar
            else:
                raise RuntimeError("invalid train_mode", self.train_mode)

    

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.compiled_metrics.update_state(targets, reconstruction)

        results = {m.name: m.result() for m in self.metrics}
        results.update({
            "loss": self.total_loss_tracker.result(),
            "prediction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        })
        return results

    def test_step(self, data):

        data, targets = data

        _, _, z = self.encoder(data, training=False)
        reconstruction = self.decoder(z, training=False)

        reconstruction_loss = tf.reduce_mean(
            keras.losses.categorical_crossentropy(targets, reconstruction)
        )

        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.compiled_metrics.update_state(targets, reconstruction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
        })
        return results

# Function Description:
# Used while Pretraining.
# Builds model and returns data allocated for pretraining of this model.
def build_model_data(
        name, x_train, y_train, x_test, y_test,
        model_i_classes, num_classes,
        input_shape, confidence_threshold
    ):
    x_train_i, y_train_i, x_test_i, y_test_i = datafilter(x_train, y_train, x_test, y_test, model_i_classes)
    y_train_i = keras.utils.to_categorical(y_train_i, num_classes)
    y_test_i = keras.utils.to_categorical(y_test_i, num_classes)
    model_i = VAE(input_shape, confidence_threshold=confidence_threshold, latent_dim=512)
    # optimizer = tf.keras.optimizers.Adam(learning_rate = 0.00004, weight_decay=0.28) # imgnet
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.000006) # cifar
    model_i.compile(optimizer=optimizer, metrics=["accuracy"])
    return x_train_i, y_train_i, x_test_i, y_test_i, model_i

# Function Description:
# Pretrains model.
def pretrain_model(
        name, model,
        batch_size, epochs,
        x_train, y_train,
        x_test, y_test,
        weights=None
    ):
    print("\n")
    print("training model {}".format(name))
    if weights is not None:
        model.set_weights(weights)
    model.train_mode = 0

    confidences_supervised = np.ones(len(y_train), dtype=np.float32)  # supervised: complete confidence in these samples
    # Shuffle the data
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_train_shuffled = x_train[indices]
    y_train_shuffled = y_train[indices]
    confidences_supervised_shuffled = confidences_supervised[indices]

    # Split the data into training and validation sets
    split_index = int(0.8 * len(x_train))  # 80% for training, 20% for validation
    x_train_new = x_train_shuffled[:split_index]
    y_train_new = y_train_shuffled[:split_index]
    confidences_train_new = confidences_supervised_shuffled[:split_index]

    x_val = x_train_shuffled[split_index:]
    y_val = y_train_shuffled[split_index:]

    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train_new, confidences_train_new, y_train_new))
    train_dataset = train_dataset.shuffle(len(x_train_new)).batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)
    target_val_accuracy_callback = TargetValAccuracyCallback(0.75, 0.85)
    
    model.fit(
    train_dataset,
    batch_size=batch_size, epochs=100, shuffle=True,
    validation_data=val_dataset, verbose=2,
    callbacks=[target_val_accuracy_callback]
    )
    model.evaluate(x_test, y_test, verbose=2)

    # prepare for online learning
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.000006)#b7
    model.compile(optimizer=optimizer, metrics=["accuracy"])

# Function Description:
# Initializes memory for each model,
# given pretraining data of model and maximum limit of samples per class in memory.
def init_memories(
        model_i, x_train_i, y_train_i, num_memorize
    ):
    memorize = datafilter_perclass(
        x_train_i, y_train_i, num_memorize,
        model_i.memorized_targets, True
    )
    model_i.memorized_data.extend(memorize[0])
    model_i.memorized_targets.extend([np.expand_dims(m.argmax(0),-1) for m in memorize[1]])
    model_i.memorized_prediction = np.ones(len(memorize[0]), dtype=np.float32)*0.9