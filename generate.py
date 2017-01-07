import argparse
import numpy as np
from data import combine_and_save
from model import generator_model

def generate_and_save(filename):
	generator = generator_model()
	noise = np.random.uniform(-1,1, (225, 100))
	generator = generator_model()
	generator.load_weights('generator_5100')
	generated_images = generator.predict(noise, verbose=1)
	combine_and_save(generated_images, filename)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('filename')
	args = parser.parse_args()
	generate_and_save(args.filename)

if __name__ == '__main__': main()

