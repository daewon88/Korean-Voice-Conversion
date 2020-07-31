import os
import pydot
import graphviz
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


class CycleGAN:
    def __init__(self, num_features, n_frames):
        self.num_features = num_features
        self.n_frames = n_frames
        self.compile_models()

    def build_generator(self):

        def down_sampling(l, filters, kernel_size, strides):
            l1 = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(l)
            l1 = tfa.layers.InstanceNormalization(epsilon=1e-6)(l1)
            l2 = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(l)
            l2 = tfa.layers.InstanceNormalization(epsilon=1e-6)(l2)
            l2 = tf.keras.activations.sigmoid(l2)
            l = tf.keras.layers.Multiply()([l1, l2])
            return l

        def residual_block(l,filters, kernel_size, strides):
            l1 = down_sampling(l,filters=filters,kernel_size=kernel_size,strides=strides)
            l1 = tf.keras.layers.Conv1D(filters=filters//2, kernel_size=kernel_size, strides=strides, padding='same')(l1)
            l1 = tfa.layers.InstanceNormalization(epsilon=1e-6)(l1)
            l = tf.keras.layers.Add()([l,l1])
            return l

        def up_sampling(l, filters, kernel_size, strides, scaling):
            l1 = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(l)
            batch = tf.shape(l1)[0]
            width = tf.shape(l1)[1]*scaling
            channel = l1.shape[2]//scaling
            l1 = tf.reshape(tensor=l1,shape=(batch,width,channel))
            l1 = tfa.layers.InstanceNormalization(epsilon=1e-6)(l1)

            l2 = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(l)
            l2 = tf.reshape(tensor=l2, shape=(batch,width,channel))
            l2 = tfa.layers.InstanceNormalization(epsilon=1e-6)(l2)
            l2 = tf.keras.activations.sigmoid(l2)
            l = tf.keras.layers.Multiply()([l1, l2])
            return l

        input = tf.keras.layers.Input(shape=(None,self.num_features))
        l1 = tf.keras.layers.Conv1D(filters=128,kernel_size=15,strides=1, padding='same')(input)
        l2 = tf.keras.layers.Conv1D(filters=128,kernel_size=15,strides=1, padding='same')(input)
        l2 = tf.keras.activations.sigmoid(l2)
        l = tf.keras.layers.Multiply()([l1, l2])

        l = down_sampling(l,filters=256, kernel_size=5, strides=2)
        l = down_sampling(l,filters=512, kernel_size=5, strides=2)

        l = residual_block(l,filters=1024, kernel_size=3, strides=1)
        l = residual_block(l,filters=1024, kernel_size=3, strides=1)
        l = residual_block(l,filters=1024, kernel_size=3, strides=1)
        l = residual_block(l,filters=1024, kernel_size=3, strides=1)
        l = residual_block(l,filters=1024, kernel_size=3, strides=1)
        l = residual_block(l,filters=1024, kernel_size=3, strides=1)

        l = up_sampling(l,filters=1024, kernel_size=5, strides=1, scaling=2)
        l = up_sampling(l,filters=512, kernel_size=5, strides=1, scaling=2)
        output = tf.keras.layers.Conv1D(filters=24,kernel_size=15, strides=1, padding='same')(l)

        model = tf.keras.Model(input,output)
        return model

    def build_discriminator(self):

        def down_sampling(l, filters, kernel_size, strides):
            l1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(l)
            l1 = tfa.layers.InstanceNormalization(epsilon=1e-6)(l1)
            l2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(l)
            l2 = tfa.layers.InstanceNormalization(epsilon=1e-6)(l2)
            l2 = tf.keras.activations.sigmoid(l2)
            l = tf.keras.layers.Multiply()([l1, l2])
            return l

        input = tf.keras.layers.Input(shape=(self.n_frames,self.num_features,1))
        l1 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=[2,1], padding='same')(input)
        l2 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=[2,1], padding='same')(input)
        l2 = tf.keras.activations.sigmoid(l2)
        l = tf.keras.layers.Multiply()([l1, l2])

        l = down_sampling(l,256,kernel_size=3, strides=2)
        l = down_sampling(l,512,kernel_size=3, strides=2)
        l = down_sampling(l,1024,kernel_size=[3,6], strides=[2,1])
        output = tf.keras.layers.Dense(1,activation='sigmoid')(l)
        model = tf.keras.Model(input,output)
        return model

    def compile_models(self):

        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()
        self.d_A.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0001, beta_1 = 0.5), loss='mse')
        self.d_B.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0001, beta_1 = 0.5), loss='mse')
        self.d_A.trainable = False
        self.d_B.trainable = False

        self.g_AB = self.build_generator()
        self.g_BA = self.build_generator()
        vocal_A = tf.keras.layers.Input(shape=(self.n_frames, self.num_features), batch_size=1)
        vocal_B = tf.keras.layers.Input(shape=(self.n_frames, self.num_features), batch_size=1)
        fake_A = self.g_BA(vocal_B)
        fake_B = self.g_AB(vocal_A)

        valid_A = self.d_A(tf.expand_dims(fake_A,-1))
        valid_B = self.d_B(tf.expand_dims(fake_B,-1))
        cycle_A = self.g_BA(fake_B)
        cycle_B = self.g_AB(fake_A)
        identity_A = self.g_BA(vocal_A)
        identity_B = self.g_AB(vocal_B)

        self.combined_model = tf.keras.Model(inputs=[vocal_A, vocal_B],
                                             outputs=[valid_A,valid_B,cycle_A,cycle_B,identity_A,identity_B])
        self.combined_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0002, beta_1 = 0.5), 
                                    loss=['mse','mse','mae','mae','mae','mae'],loss_weights=[1, 1, 10, 10, 5, 5])

    def train(self, vocal_A, vocal_B, batch_size=1):
        valid = np.ones(shape=(batch_size, 8, 6))
        fake = np.zeros(shape=(batch_size, 8, 6))
        dA_loss, dB_loss = self.train_discriminator(vocal_A,vocal_B,valid,fake)
        generator_loss = self.train_generator(vocal_A,vocal_B,valid)
        return (dA_loss, dB_loss, generator_loss)

    def train_discriminator(self, vocal_A, vocal_B, valid, fake):
        fake_A = self.g_BA(vocal_B)
        fake_B = self.g_AB(vocal_A)

        fake_A = tf.expand_dims(fake_A, -1)
        fake_B = tf.expand_dims(fake_B, -1)
        vocal_A = tf.expand_dims(vocal_A, -1)
        vocal_B = tf.expand_dims(vocal_B, -1)
        
        input_A  = np.concatenate((vocal_A,fake_A),axis=0)
        input_B  = np.concatenate((vocal_B,fake_B),axis=0) 
        answer = np.concatenate((valid,fake),axis=0)
        dA_loss = self.d_A.train_on_batch(input_A,answer)
        dB_loss = self.d_B.train_on_batch(input_B,answer)
        
        '''
        dA_loss_real = self.d_A.train_on_batch(vocal_A, valid)
        dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
        dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

        dB_loss_real = self.d_B.train_on_batch(vocal_B, valid)
        dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
        dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)
        '''
        
        return (dA_loss, dB_loss)
        

    def train_generator(self,vocal_A,vocal_B,valid):

        return self.combined_model.train_on_batch([vocal_A,vocal_B],[valid,valid,vocal_A,vocal_B,vocal_A,vocal_B])

    def save(self,model_dir):
        self.d_A.save(model_dir + '/d_A')
        self.d_B.save(model_dir + '/d_B')
        self.g_AB.save(model_dir + '/g_AB')
        self.g_BA.save(model_dir + '/g_BA')
        self.combined_model.save(model_dir + '/combined_model')

    def load(self,model_dir):
        self.d_A = tf.keras.models.load_model(model_dir + '/d_A')
        self.d_B = tf.keras.models.load_model(model_dir + '/d_B')
        self.g_AB = tf.keras.models.load_model(model_dir + '/g_AB')
        self.g_BA = tf.keras.models.load_model(model_dir + '/g_BA')
        self.combined_model = tf.keras.models.load_model(model_dir + '/combined_model')


    def test(self, inputs, direction):
        if direction == 'A2B':
            return self.g_AB(inputs)
        if direction == 'B2A':
            return self.g_BA(inputs)
'''
os.environ["PATH"] = os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
tf.keras.utils.plot_model(model)
'''













