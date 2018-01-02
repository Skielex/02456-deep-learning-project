# Import modules.
import os
import datetime
import numpy as np

from keras import backend as K
from keras import metrics
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.layers import Input, Dense, Lambda, Layer, RepeatVector, Reshape
from keras.layers.merge import Concatenate, concatenate
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.losses import binary_crossentropy, kullback_leibler_divergence
from keras.utils import to_categorical

class ExtendedModel(Model):
    """Class adding some extras to Keras Model class.
    """

    def __init__(self, inputs, outputs, name='Model', tensorboard=False, log_dir='tf_logs', checkpoint=0, checkpoint_dir='models'):
        # Get current timestamp.
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        output_dir = 'output'
        self.base_dir = os.path.join(output_dir, name, timestamp)

        # Create default tensorboard callback.
        if tensorboard:
            self.tensorboard_callback = TensorBoard(log_dir=os.path.join(self.base_dir, log_dir, timestamp),
                # write_grads=True,
                histogram_freq=25,
                write_images=False)

        # Create default checkpoint callback.
        if checkpoint > 0:

            checkpoint_dir = os.path.join(self.base_dir, checkpoint_dir)

            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            self.checkpoint_callback = ModelCheckpoint(os.path.join(checkpoint_dir, timestamp + '.weights.{epoch:05d}-{val_loss:.2f}.hdf5'),
                monitor='val_loss',
                verbose=0,
                save_best_only=True,
                save_weights_only=False,
                mode='auto',
                period=checkpoint)

        # Init Keras model.
        super(ExtendedModel, self).__init__(inputs, outputs, name=name)

    def instantiate_layers(self, in_layer, layers):
        """Function for instantiating a list of layers. List may be nested.

            # Arguments
                in_layer: The layer to use as input layer.
                layers: A list of layers to initialize.

            # Returns
                The final layer (output).
        """

        l = in_layer
        for layer in layers:
            # print(l.shape)
            if isinstance(layer, list):
                l = self.instantiate_layers(l, layer)
            else:
                l = layer(l)
        return l

    def save_summary(self, models=None):
        """Saves the summary to a text file.
        """

        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        print_fn = lambda s: summary_file.write(s + '\n')

        with open(os.path.join(self.base_dir, 'summary.txt'), 'w') as summary_file:
            self.summary(print_fn=print_fn)

            if models:
                for model in models:
                    model.summary(print_fn=print_fn)

class Encoder(ExtendedModel):
    """An encoder.
    """

    def __init__(self, input_shape, encoder_layers, input_name='encoder_input', model_name='Encoder'):

        # Create input.
        x = Input(shape=input_shape, name=input_name)

        # Instantiate layers.
        with K.name_scope(model_name):
            x_encoded = self.instantiate_layers(x, encoder_layers)

        # Initialize super.
        super(Encoder, self).__init__(x, x_encoded, name=model_name)

class Decoder(ExtendedModel):
    """A decoder.
    """

    def __init__(self, input_shape, decoder_layers, input_name='decoder_input', model_name='Decoder'):

        # Create input.
        x = Input(shape=input_shape, name=input_name)

        # Instantiate layers.
        with K.name_scope(model_name):
            x_decoded = self.instantiate_layers(x, decoder_layers)

        # Create model.
        super(Decoder, self).__init__(x, x_decoded, name=model_name)

class Autoencoder(ExtendedModel):
    """A basic autoencoder.
    """

    def __init__(self, input_shape, encoder_layers, decoder_layers, input_name='autoencoder_input', model_name='Autoencoder', optimizer='adam', loss='binary_crossentropy', log_dir='tf_logs', checkpoint_dir='models', compile_model=True):

        # Create input.
        x = Input(shape=input_shape, name=input_name)

        # Create encoder and decoder.
        self.encoder = Encoder(input_shape, encoder_layers)
        self.decoder = Decoder(self.encoder.layers[-1].output_shape[1:], decoder_layers)

        # Instantiate layers.
        with K.name_scope(model_name):
            x_encoded = self.encoder(x)
            x_decoded = self.decoder(x_encoded)

        # Create model.
        super(Autoencoder, self).__init__(x, x_decoded, name=model_name, tensorboard=True, log_dir=log_dir, checkpoint=10, checkpoint_dir=checkpoint_dir)

        if compile_model:
            # Compile model.
            self.compile(optimizer=optimizer, loss=loss)

class VariationalLayer(Layer):
    """A custom variational layer.
    """

    def __init__(self, num_latent, z_mean_layer, z_log_var_layer, **kwargs):
        self.num_latent = num_latent

        self.z_mean_layer = z_mean_layer
        self.z_log_var_layer = z_log_var_layer

        self.kl_losses = []

        super(VariationalLayer, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):

        # Pass previous layer to both mean and varriance.
        z_mean = inputs[0]
        z_log_var = inputs[1]

        input_shape = K.shape(z_mean)
        # eps = K.random_normal(shape=(batch_size, self.num_samples, self.num_latent), mean=0.0, stddev=1.0)
        eps = K.random_normal(shape=input_shape, mean=0.0, stddev=1.0)

        # Reparameterisation trick: Sample z = mu + sigma*epsilon
        #z = K.expand_dims(z_mean, axis=1) + (K.exp(0.5 * K.expand_dims(z_log_var, 1)) * eps)
        z = z_mean + K.exp(0.5 * z_log_var) * eps

        # Add loss to layer.
        self.kl_losses.append(self.kl_loss(z_mean, z_log_var))
        # self.add_loss(kl_loss, inputs=inputs)

        return z

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = input_shape[0]
        return output_shape

    def kl_loss(self, z_mean, z_log_var):
        """Computes the KL-loss.
        """
        input_shape = K.shape(z_mean)
        z_mean = K.reshape(z_mean, (input_shape[0], -1))
        z_log_var = K.reshape(z_log_var, (input_shape[0], -1))
        return -0.5 * K.sum(1. + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

class VariationalAutoencoder(ExtendedModel):
    """A variational autoencoder.
    """

    def __init__(self, input_shape, encoder_layers, decoder_layers, input_name='vae_input', model_name='VAE', optimizer='adam', log_dir='tf_logs', checkpoint_dir='models', compile_model=True, z_dim=2):

        # Calculate full size.
        data_size = np.product(input_shape)

        # Create input.
        x = Input(shape=input_shape, name=input_name)

        # Define the latent parameters as two hidden layers.
        # Mean of q(z|x).
        z_mean_layer = Dense(z_dim, name='z_mean')

        # Log varriance of q(z|x). Clipping from VAE exercise.
        z_log_var_layer = Dense(z_dim, name='z_log_var', activation=lambda x: K.clip(x,-10,10))

        # Latent sampling z-layer.
        z_layer = VariationalLayer(z_dim, z_mean_layer, z_log_var_layer, name='z')

        # Add z-layer.
        encoder_layers += [
            lambda l: [z_mean_layer(l), z_log_var_layer(l)],
            z_layer
        ]

        # Create encoder and decoder.
        self.encoder = Encoder(input_shape, encoder_layers)
        self.decoder = Decoder(self.encoder.layers[-1].output_shape[1:], decoder_layers)

        # Instantiate layers.
        with K.name_scope(model_name):
            z = self.encoder(x)
            x_decoded = self.decoder(z)

        # Create model.
        super(VariationalAutoencoder, self).__init__(x, x_decoded, name=model_name, tensorboard=True, log_dir=log_dir, checkpoint=10, checkpoint_dir=checkpoint_dir)

        # Loss functions.
        def log_p_x_given_z(x, x_decoded):
            # The cross extropy loss here is also -log(p(x|z)).
            xent_loss = -data_size * metrics.binary_crossentropy(x, x_decoded)
            return K.mean(xent_loss)

        def kl_qp(x, x_decoded):
            # The KL-loss is defined in the z-layer.
            return K.mean(z_layer.kl_losses[-1])

        def vae_loss(x, x_decoded):
            # The variational lower bound or evidence lower bound objective (ELBO).
            elbo = log_p_x_given_z(x, x_decoded) - kl_qp(x, x_decoded)
            return -elbo

        if compile_model:
            # Compile model.
            self.compile(optimizer=optimizer, loss=vae_loss, metrics=[log_p_x_given_z, kl_qp])

class SemiSupervisedClassifier(ExtendedModel):
    """A semi-supervised classifier using a variational autoencoder.
    """

    def __init__(self, input_shape, encoder_layers, decoder_layers, classifier_layers, num_classes, factor_labels, model_name='SSClassifier', optimizer='adam', log_dir='tf_logs', checkpoint_dir='models', compile_model=True, z_dim=2):

        # Calculate full size. Used for log_px calculation, as the sum of rows, columns and channels.
        data_size = np.product(input_shape)

        # Create new inputs.
        l_in_x = Input(shape=input_shape, name='l_in_x')
        l_in_y = Input(shape=(num_classes,), name='l_in_y')

        # # Instantiate classifier layers.
        with K.name_scope('Classifier'):
            l_y = self.instantiate_layers(l_in_x, classifier_layers)
            self.classifier = Model(l_in_x, l_y, name='Classifier')

        # Define the latent parameters as two hidden layers.
        # Mean of q(z|x).
        z_mean_layer = Dense(z_dim, name='z_mean')

        # Log varriance of q(z|x). Clipping from VAE exercise.
        z_log_var_layer = Dense(z_dim, name='z_log_var', activation=lambda x: K.clip(x,-10,10))

        # Latent sampling z-layer.
        l_z = VariationalLayer(z_dim, z_mean_layer, z_log_var_layer, name='z')

        l_muq = l_z.z_mean_layer
        l_logvaeq = l_z.z_log_var_layer

        def inject(layers):
            for i, layer in enumerate(layers):
                if isinstance(layer, list):
                    if inject(layer):
                        return True
                if isinstance(layer, Dense):
                    layers.insert(i, lambda l: concatenate([l, l_in_y]))
                    return True
            return False

        # Inject concatenation before first dense layer.
        inject(encoder_layers)

        # Add special layers.
        encoder_layers += [
            lambda l: [z_mean_layer(l), z_log_var_layer(l)],
            l_z
        ]

        # Instantiate encoder.
        with K.name_scope('Encoder'):
            z_label = self.instantiate_layers(l_in_x, encoder_layers)
            self.encoder = Model([l_in_x, l_in_y], z_label, name='Encoder')

        encoder_output_shape = self.encoder.layers[-1].output_shape[1:]
        l_in_z = Input(shape=encoder_output_shape, name='l_in_z') 

        # Reshape y if necessary.
        l_in_y_reshaped = l_in_y
        if len(encoder_output_shape) > 2:
            l_in_y_reshaped = Lambda(lambda l: K.tile(K.expand_dims(K.expand_dims(l, axis=1), axis=1), [1] + list(encoder_output_shape[:-1]) + [1]))(l_in_y)
            # l_in_y_reshaped = Reshape((1,1,num_classes))(l_in_y)
                

        # Instantiate decoder.
        with K.name_scope('Decoder'):
            l_mux = self.instantiate_layers(concatenate([l_in_z, l_in_y_reshaped]), decoder_layers)
            self.decoder = Model([l_in_z, l_in_y], l_mux, name='Decoder')

        # New input variables to use for combined.
        sym_x_l = Input(shape=input_shape, name='x_l')
        sym_x  = Input(shape=input_shape, name='x')
        sym_y  = Input(shape=(num_classes,), name='y')

        with K.name_scope('Labeled'):
            # Instantiate classifier layers.
            y_train_l = self.classifier(sym_x_l)

            # Instantiate encoder.
            z_train_l = self.encoder([sym_x_l, sym_y])

            # Instantiate decoder.
            mux_train_l = self.decoder([z_train_l, sym_y])

        # Labeled loss.
        def log_px_l(x, y): return -data_size * metrics.binary_crossentropy(K.reshape(sym_x_l, (K.shape(sym_x_l)[0], -1)), K.reshape(mux_train_l, (K.shape(sym_x_l)[0], -1)))
        def KL_qp_l(x, y): return l_z.kl_losses[-2] # Sencond to last loss added.
        def log_qy_l(x, y): return -K.categorical_crossentropy(sym_y, y_train_l)
        # def log_qy_l(x, y): return K.sum(sym_y * K.log(y_train_l+1e-8), axis=1)
        def py_l(x, y): return K.softmax(K.reshape(K.tile(K.zeros((num_classes,)), K.shape(sym_x_l)[0:1]), (-1, num_classes)))
        # def py_l(x, y): return K.softmax(K.zeros((batch_size, num_classes)))
        def log_py_l(x, y): return -K.categorical_crossentropy(py_l(x, y), sym_y)
        alpha = 0.1*factor_labels
        def LL_l_eval(x, y): return K.mean(log_px_l(x, y) + log_py_l(x, y) - KL_qp_l(x, y))
        def LL_l(x, y): 
            with K.name_scope('l_loss'):
                return K.mean(log_px_l(x, y) + log_py_l(x, y) - KL_qp_l(x, y) + alpha * log_qy_l(x, y))

        # Special variables.
        def _t(inputs):
            _t_eye = K.eye(num_classes)
            _t_u = K.reshape(_t_eye, (num_classes, 1, num_classes))
            _t_u = K.tile(_t_u, [1, K.shape(inputs)[0], 1])
            # _t_u = K.repeat_elements(_t_u, batch_size, axis=1)
            _t_u = K.reshape(_t_u, (-1, num_classes))
            return _t_u

        t_u = Lambda(_t, name='t_u')(sym_x)

        def _x(inputs):
            _x_u = K.expand_dims(inputs, axis=0)
            # _x_u = K.reshape(inputs, (1, batch_size) + input_shape)
            _x_u = K.repeat_elements(_x_u, num_classes, axis=0)
            _x_u = K.reshape(_x_u, (-1,) + input_shape)
            return _x_u

        x_u = Lambda(_x, name='x_u')(sym_x)

        with K.name_scope('Unlabeled'):
            # Instantiate classifier layers.
            y_train = self.classifier(sym_x)

            # Instantiate encoder.
            z_train = self.encoder([x_u, t_u])    

            # Instantiate decoder.
            mux_train = self.decoder([z_train, t_u])

        # Unlabeled loss.
        def log_px_given_z_u(x, y): return -data_size * metrics.binary_crossentropy(K.reshape(x_u, (K.shape(x_u)[0],-1)), K.reshape(mux_train, (K.shape(x_u)[0], -1)))
        def KL_qp_u(x, y): return l_z.kl_losses[-1] # Last added.
        def py_u(x, y): return K.softmax(K.reshape(K.tile(K.zeros((num_classes,)), (K.shape(x_u)[0:1])), (K.shape(x_u)[0], num_classes)))
        # def py_u(x, y): return K.softmax(K.zeros((K.int_shape(x_u)[0], num_classes)))
        def log_py_u(x, y): return -K.categorical_crossentropy(py_u(x, y), t_u)
        def LL_u(x, y): 
            with K.name_scope('u_loss'):
                _LL_u = log_px_given_z_u(x, y) + log_py_u(x, y) - KL_qp_u(x, y)
                _LL_u = K.transpose(K.reshape(_LL_u, (num_classes, -1)))
                return K.mean(K.sum(y_train * (_LL_u - K.log(y_train + K.epsilon())), axis=-1))

        def LL(x, y): 
            with K.name_scope('t_loss'):
                return -(LL_u(x, y) + LL_l(x, y))
        
        # def eval_acc(x, y): return K.mean(metrics.categorical_accuracy(y_eval, sym_y))
        def acc(x, y): return K.mean(metrics.categorical_accuracy(y_train_l, sym_y))

        class SemiSupervisedCostLayer(Layer):
            def __init__(self, **kwargs):
                self.is_placeholder = True
                super(SemiSupervisedCostLayer, self).__init__(**kwargs)

            def call(self, inputs):
                x = inputs[0]
                
                # loss = LL(0,0)
                # self.add_loss(loss, inputs=inputs)
                # We don't use this output.
                return x

        cost_layer = SemiSupervisedCostLayer(name='cost_layer')([mux_train, mux_train_l, y_train, y_train_l])

        # Create this model.
        super(SemiSupervisedClassifier, self).__init__(
            inputs=[sym_x, sym_x_l, sym_y],
            outputs=cost_layer,
            tensorboard=True,
            log_dir=log_dir,
            name=model_name,
            checkpoint=5            
        )

        # def mean(f):
        #     return lambda x, y: K.mean(f(x, y))

        if compile_model:
            # Compile model.
            self.compile(optimizer=optimizer, loss=LL, metrics=[
                acc,
                LL_u, log_py_u, py_u, KL_qp_u, log_px_given_z_u,
                LL_l, LL_l_eval, log_py_l, py_l, log_qy_l, KL_qp_l, log_px_l
            ])