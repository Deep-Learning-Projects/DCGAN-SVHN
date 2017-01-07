import numpy as np
from keras.optimizers import Adam
from data import *
from model import *

def train():
	discriminator = discriminator_model()
	generator = generator_model()
	discriminator_on_generator = generator_containing_discriminator(generator, discriminator)
	d_optimizer = Adam(lr='0.0002')
	g_optimizer = Adam(lr='0.0002')
	generator.compile(loss='binary_crossentropy', optimizer='SGD')
	discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=g_optimizer)
	discriminator.trainable = True
	discriminator.compile(loss='binary_crossentropy', optimizer=d_optimizer)
	n_epochs = 15000
	for epoch in range(n_epochs):
		for i in range(num_batches):	
			noise = np.random.uniform(-1,1,(batch_size, 100))
			image_batch = get_batch(i)
			generated_images = generator.predict(noise, verbose=0)
			if i == 0:
				combine_and_save(generated_images,'images/'+str(epoch)+'.png')
			X = np.concatenate((image_batch, generated_images))
			y = [1] * batch_size + [0] * batch_size
			d_loss = discriminator.train_on_batch(X, y)
			print('epoch %d, batch %d, d_loss: %f' % (epoch, i, d_loss))
			noise = np.random.uniform(-1,1,(batch_size, 100))
			discriminator.trainable = False
			g_loss = discriminator_on_generator.train_on_batch(noise,[1]*batch_size)
			discriminator.trainable = True
			print('epoch %d, batch %d, g_loss: %f' % (epoch, i, g_loss))
		if epoch % 50 == 0:
			generator.save_weights('weights/generator_'+str(epoch), True)
			discriminator.save_weights('weights/discriminator_'+str(epoch), True)


if __name__ == '__main__': train()
