from keras.models import Sequential
from keras.layers import Dense, Reshape, Activation, BatchNormalization
from keras.layers.convolutional import UpSampling2D, Convolution2D, MaxPooling2D
from keras.layers.core import Flatten


def generator_model():
	model = Sequential()
	model.add(Dense(input_dim=100, output_dim=1024))
	model.add(Activation('tanh'))

	model.add(Dense(128*4*4))
	model.add(BatchNormalization())
	model.add(Activation('tanh'))

	model.add(Reshape((128,4,4), input_shape=(128*4*4,)))
	model.add(UpSampling2D(size=(2,2)))
	model.add(Convolution2D(64,5,5, border_mode='same'))
	model.add(BatchNormalization())
	model.add(Activation('tanh'))

	model.add(UpSampling2D(size=(2,2)))
	model.add(Convolution2D(32,5,5, border_mode='same'))
	model.add(BatchNormalization())
	model.add(Activation('tanh'))

	model.add(UpSampling2D(size=(2,2)))
	model.add(Convolution2D(3,5,5, border_mode='same'))
	model.add(Activation('tanh'))
	
	return model

def discriminator_model():
	model = Sequential()
	model.add(Convolution2D(64,5,5, border_mode='same', input_shape=(3,32,32)))
	model.add(Activation('tanh'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	
	model.add(Convolution2D(128,5,5))
	model.add(Activation('tanh'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	
	model.add(Flatten())
	model.add(Dense(1024))
	model.add(Activation('tanh'))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))
	
	return model

def generator_containing_discriminator(generator, discriminator):
	model = Sequential()
	model.add(generator)
	discriminator.trainable = False
	model.add(discriminator)
	return model
