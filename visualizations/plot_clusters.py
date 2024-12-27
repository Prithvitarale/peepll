import pdb
import random
random.seed(1)
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# exit()
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
# from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import (
    Dense, Input 
)

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
class VAE(keras.Model):
    def __init__(self, input_shape, latent_dim=128, encoder=None, decoder=None, confidence_threshold=0.5, **kwargs):
        super(VAE, self).__init__(**kwargs)

        self.confidence_threshold = confidence_threshold

        if encoder is None:
            inputs = keras.Input(shape=(input_shape))
            # change model here
            vgg = VGG16(weights='imagenet', include_top=False)
            inputs = preprocess_input(inputs)
            x = vgg(inputs)
            # x = tf.reshape(x, [-1, x.shape[-1]])
            x = tf.reshape(x, [-1, x.shape[1]*x.shape[2]*x.shape[3]])

            z_mean = keras.layers.Dense(latent_dim)(x)
            z_log_var = keras.layers.Dense(latent_dim)(x)
            z = Sampling()([z_mean, z_log_var])
            self.encoder = keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")
        else:
            self.encoder = encoder

        if decoder is None:
            inputs = keras.Input(shape=(latent_dim,))
            # x = keras.layers.Dropout(0.1)(inputs)
            # x = keras.layers.Dropout(0.4)(inputs)
            x = keras.layers.Dropout(0)(inputs)
            # tf.print("Dropout layer output (should be mostly zeros):", x)
            # x = Dense(128, activation='relu')(x) not this
            o = x
            o = keras.layers.Dense(256, activation='relu')(o)
            # o = keras.layers.Dropout(0.4)(o)
            o = keras.layers.Dense(128)(o)
            o = keras.layers.Dense(100, activation="softmax", name = "classifier")(o)
            self.decoder = keras.Model(inputs, [o], name="decoder")
        else:
            self.decoder = decoder

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        # self.generative_loss_tracker = keras.metrics.Mean(
        #     name="generative_loss"
        # )
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
            # self.generative_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, data, training=True):
        z_mean, z_log_var, z = self.encoder(data, training=training)
        classification = self.decoder(z, training=training)
        # r = tf.convert_to_tensor([classification, confidence, reconstruction, z, z_mean, z_log_var])
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
                total_loss = reconstruction_loss + tf.reduce_mean(0.007*(kl_loss))
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

def datafilter(X_train, Y_train, X_test, Y_test, classes):
    if np.shape(Y_train)[1] == 1:
        train_mask = np.isin(Y_train[:, 0], classes)
        test_mask = np.isin(Y_test[:, 0], classes)
    elif np.shape(Y_train)[1] > 1:
        train_mask = np.isin(Y_train.argmax(1), classes)
        test_mask = np.isin(Y_test.argmax(1), classes)
    else:
        train_mask = np.isin(Y_train, classes)
        test_mask = np.isin(Y_test, classes)
    print("datafilter classes", classes)
    X_train, Y_train = X_train[train_mask], Y_train[train_mask]
    X_test, Y_test = X_test[test_mask], Y_test[test_mask]
    return (X_train, Y_train, X_test, Y_test)

def build_model_data(
        name, x_train, y_train, x_test, y_test,
        model_i_classes, num_classes,
        input_shape, confidence_threshold
    ):
    x_train_i, y_train_i, x_test_i, y_test_i = datafilter(x_train, y_train, x_test, y_test, model_i_classes)
    y_train_i = keras.utils.to_categorical(y_train_i, num_classes)
    y_test_i = keras.utils.to_categorical(y_test_i, num_classes)
    model_i = VAE(input_shape, confidence_threshold=confidence_threshold, latent_dim=512)
    optimizer = tf.keras.optimizers.AdamW(learning_rate = 0.000006, weight_decay=0.0007)#b7
    model_i.compile(optimizer=optimizer, metrics=["accuracy"])
    # for confidence we evaluate on datapoints 300:
    # return x_train_i[:300], y_train_i[:300], x_test_i[:300], y_test_i[:300], model_i, x_train_i[300:], y_train_i[300:], x_test_i[300:], y_test_i[300:],
    return x_train_i, y_train_i, x_test_i, y_test_i, model_i

def separate_pretrain_eval(x_train, y_train, x_test, y_test, classes):
    xtr, ytr, xtr_eval, ytr_eval = None,None,None,None 
    y_train = np.argmax(y_train, 1)
    for cl in classes:
        c_idxs = np.where(y_train==cl)[0]
        if xtr is None:
            xtr = x_train[c_idxs][:300]
            ytr = y_train[c_idxs][:300]
            xtr_eval = x_train[c_idxs][300:]
            ytr_eval = y_train[c_idxs][300:]
        else:
            # pdb.set_trace()
            xtr = np.append(xtr, x_train[c_idxs][:300], 0)
            ytr = np.append(ytr, y_train[c_idxs][:300], 0)
            xtr_eval = np.append(xtr_eval, x_train[c_idxs][300:], 0)
            ytr_eval = np.append(ytr_eval, y_train[c_idxs][300:], 0)
    return xtr, ytr, xtr_eval, ytr_eval

def datafilter_perclass(X_train, Y_train, num_per_class, classes, negate=False):

    if np.shape(Y_train)[1] == 1:
        Y_train_1d = Y_train[:, 0]
    elif np.shape(Y_train)[1] > 1:
        Y_train_1d = Y_train.argmax(1)
    else:
        Y_train_1d = Y_train
    train_mask = np.isin(Y_train_1d, classes)

    if negate:
        train_mask = ~train_mask

    Y_train_1d = Y_train_1d[train_mask]
    X_train, Y_train = X_train[train_mask], Y_train[train_mask]

    X_new = []
    Y_new = []
    avail_classes, avail_idxs = np.unique(Y_train_1d, return_inverse=True)
    print("Adding samples for classes", avail_classes)
    for idx, cls in enumerate(avail_classes):
        cls_mask = avail_idxs == idx
        X_new.extend(X_train[cls_mask][:num_per_class])
        Y_new.extend(Y_train[cls_mask][:num_per_class])

    return np.concatenate([X_new], 0), np.concatenate([Y_new], 0)

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


def identify_from_memory_scalable_forkl(model, queries, p_batch, confN, ms_scores):
    score_for_input = []
    confidences = []
    pred = np.argmax(p_batch)
    queries = np.expand_dims(queries, 0)

    memory = np.concatenate([model.memorized_data], 0)
    memory = tf.squeeze(memory, axis=4)
    _, memory_z, z_means, z_logs = model(memory) # 900, 128
    _, queries_z, q_mean, q_log = model(queries) # 128,128

    all_scores = []
    for q, query in enumerate(queries):
        matching_indices = np.isin(model.memorized_targets, pred)
        matching_indices = np.where(matching_indices)[0].flatten()
        if len(matching_indices) == 0:
            all_scores.append(0)
        else:
            matching_z_means = np.array(z_means)[matching_indices, :] # 50, 128
            matching_z_logs = np.array(z_logs)[matching_indices, :] # 50, 128
            matching_z = np.array(memory_z)[matching_indices, :] # 50, 128
            # query_var = np.exp(q_log[q, :]) # 128
            query_var = np.exp(q_log) # 128
            memory_var = np.exp(matching_z_logs) # 50, 128
            mean_memory_var = np.mean(memory_var, axis=0)
            mean_memory_mean = np.mean(matching_z_means, axis=0)

            # entropy = confN[q][0]
            entropy = confN
            # ms = ms_scores[q][0]

            # pdb.set_trace()
            uncertainty_score = np.exp(-0.01*np.sum(np.abs(mean_memory_var-query_var)))
            uncertainty_score = (uncertainty_score - 0.4)/(0.5)
            uncertainty_score = max(uncertainty_score, 0)
            uncertainty_score = min(uncertainty_score, 1)


            # mean_distance = np.exp(-0.1*tf.norm(mean_memory_mean-q_mean[q, :]))
            mean_distance = np.exp(-0.1*tf.norm(mean_memory_mean-q_mean))
            mean_distance = (mean_distance - 0.4)/(0.4)
            mean_distance = max(mean_distance, 0)
            mean_distance = min(mean_distance, 1)
            all_scores.append((mean_distance + uncertainty_score + entropy)/3)
    return tf.convert_to_tensor(all_scores, dtype=tf.float32)


model_classes = [1, 11, 21, 31, 41, 10]
input_shape = (32, 32, 3)
input_shape_b = (None,) + input_shape
figure_i = "2.0_1"

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
print("x_train shape", x_train.shape)
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
y_train_categorical = keras.utils.to_categorical(y_train, 100)
y_test_categorical = keras.utils.to_categorical(y_test, 100)


# import sys
# import os
# Add the previous directory to the system path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from data import load_dataset
# (x_train, y_train), (x_test, y_test) = load_dataset.load_data() # imagenet
# print("x_train shape", x_train.shape)
# x_train = x_train.astype("float32") / 255
# x_test = x_test.astype("float32") / 255
# x_train = np.expand_dims(x_train, -1)
# x_test = np.expand_dims(x_test, -1)
# y_train_categorical = keras.utils.to_categorical(y_train, 100)
# y_test_categorical = keras.utils.to_categorical(y_test, 100)


x_tr, y_tr, x_te, y_te, m = build_model_data(
f"{1}", x_train, y_train, x_test, y_test,
model_classes, 100,
input_shape, 0.65
)

_y_tr = y_tr
_x_tr = x_tr
x_tr, y_tr, x_tr_eval, y_tr_eval = separate_pretrain_eval(x_tr, y_tr, x_te, y_te, model_classes)

m.build(input_shape_b)
m.load_weights(f'/work/pi_hava_umass_edu/prithvi/save_cifar100/save_cifar100/1_newdatakl5_{figure_i}.h5')
# m.load_weights(f'/work/pi_hava_umass_edu/prithvi/save_cifar100/save_cifar100/1_newdatakl5_{figure_i}.h5')
# m.load_weights(f'/work/pi_hava_umass_edu/prithvi/save_mimgnet/1_img_10agentsr_5_{figure_i}.h5')
print(f'loaded model')

init_memories(m, _x_tr, _y_tr, 50)



from sklearn.manifold import TSNE
def get_latent_vectors(model, data):
    z_mean, z_log_var, z = model.encoder.predict(data)
    predictions = model.decoder.predict(z)
    return z, predictions

# Get latent vectors for the training data
latent_vectors, predictions = get_latent_vectors(m, x_tr)
latent_vectors_eval, predictions_eval = get_latent_vectors(m, x_tr_eval)
u_x_tr, u_y_tr, u_x_te, u_y_te, me = build_model_data(
f"{1}", x_train, y_train, x_test, y_test,
[5], 100,
input_shape, 0.65
)
u_labels = np.argmax(u_y_te, 1)
u_latent_vectors, u_predictions = get_latent_vectors(m, u_x_te)
tsne = TSNE(n_components=2, random_state=0)
all_latents = np.append(latent_vectors, latent_vectors_eval, 0)
all_latents_all = np.append(all_latents, u_latent_vectors, 0)
latent_2d = tsne.fit_transform(all_latents_all)
u_latent_2d = latent_2d[all_latents.shape[0]:, :]
latent_2d = latent_2d[:all_latents.shape[0], :]
# labels = np.argmax(y_tr_eval, axis=1)
# labels = y_tr_eval
# plot_latent_space_again(latent_2d, labels, np.argmax(predictions, 1))

# Reduce dimensionality to 2D using t-SNE
# tsne = TSNE(n_components=2, random_state=0)
# latent_2d = tsne.fit_transform(latent_vectors)

# Plotting function
def plot_latent_space(latent_2d, labels, tr_size, eval_size, title='TSNE Latent Space'):
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    from matplotlib.colors import ListedColormap
    # colors = ['olivedrab', 'black', 'royalblue', 'orchid', 'gold', 'teal', 'firebrick']
    colors = ['olivedrab', 'royalblue', 'orchid', 'gold', 'teal', 'firebrick']
    # colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k']
    label_color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
    color_list = [label_color_map[label] for label in labels]
    # color_list_preds = [label_color_map[label] for label in predictions]

    scatter = plt.scatter(latent_2d[:tr_size, 0], latent_2d[:tr_size, 1], c=color_list[:tr_size], alpha=0.2)
    scatter = plt.scatter(latent_2d[tr_size:, 0], latent_2d[tr_size:, 1], c=color_list[tr_size:], alpha=1, marker='+')
    # cross_indices = np.where(labels==5)[0]
    # plt.scatter(latent_2d[cross_indices, 0], latent_2d[cross_indices, 1], c=[color_list_preds[i] for i in cross_indices], marker='x', s=100, linewidths=1)
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=str(label), markersize=10,
                          markerfacecolor=color) for label, color in label_color_map.items()]
    plt.legend(handles=handles, title="Classes")
    plt.title(title)
    plt.savefig('clustersofvaeonlyknown_all2.png')

# Plot the latent space
# pdb.set_trace()
# labels = np.argmax(y_tr, axis=1)
# labels = y_tr
labels = np.append(y_tr, y_tr_eval, 0)
# print(np.argmax(predictions, 1))
# print(labels)
# print(np.argmax(predictions[np.where(labels==5)[0]]))
plot_latent_space(latent_2d, labels, latent_vectors.shape[0], latent_vectors_eval.shape[0])

def plot_latent_space_unknown(latent_2d, labels, predictions, unique_labels, title='TSNE Latent Space'):
    # plt.figure(figsize=(10, 8))
    # unique_labels = np.unique(labels)
    # from matplotlib.colors import ListedColormap
    colors = ['olivedrab', 'black', 'royalblue', 'orchid', 'gold', 'teal', 'firebrick']
    # colors = ['olivedrab', 'royalblue', 'orchid', 'gold', 'teal', 'firebrick']
    # colors = ['black']
    # colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k']
    label_color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
    color_list = [label_color_map[label] for label in labels]
    color_list_preds = [label_color_map[label] for label in predictions]

    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c='black', alpha=0.4)
    # scatter = plt.scatter(latent_2d[tr_size:, 0], latent_2d[tr_size:, 1], c=color_list[tr_size:], alpha=1, marker='+')
    cross_indices = np.where(labels==5)[0]
    plt.scatter(latent_2d[cross_indices, 0], latent_2d[cross_indices, 1], c=[color_list_preds[i] for i in cross_indices], marker='x', s=100, linewidths=1)
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=str(label), markersize=10,
                          markerfacecolor=color) for label, color in label_color_map.items()]
    plt.legend(handles=handles, title="Classes")
    plt.title(title)
    plt.savefig('clustersofvae_all2.png')



# u_latent_2d = tsne.fit_transform(u_latent_vectors)
new_labels = np.append(u_labels, labels, 0)
plot_latent_space_unknown(u_latent_2d, u_labels, np.argmax(u_predictions, 1), np.unique(new_labels))



# Confidence for predictions
total_conf = 0
for i, q in enumerate(x_tr):
    # pdb.set_trace()
    p = predictions[i]
    confN = []
    entropy = -np.sum(p * np.log2(p + 1e-10))
    entropy_mapped = np.exp(-entropy)
    confidence = identify_from_memory_scalable_forkl(m, q, p, entropy_mapped, 1)
    total_conf+=confidence
print(f"Avg confidence: {total_conf/x_tr.shape[0]}")

# Confidence for u_predictions
total_conf = 0
for i, q in enumerate(u_x_te):
    # pdb.set_trace()
    p = u_predictions[i]
    confN = []
    entropy = -np.sum(p * np.log2(p + 1e-10))
    entropy_mapped = np.exp(-entropy)
    confidence = identify_from_memory_scalable_forkl(m, q, p, entropy_mapped, 1)
    total_conf+=confidence
print(f"Avg confidence: {total_conf/u_x_te.shape[0]}")







# latent_vectors, predictions = get_latent_vectors(m, x_tr_eval)
# tsne = TSNE(n_components=2, random_state=0)
# latent_2d = tsne.fit_transform(latent_vectors)
# # labels = np.argmax(y_tr_eval, axis=1)
# labels = y_tr_eval
# plot_latent_space_again(latent_2d, labels, np.argmax(predictions, 1))