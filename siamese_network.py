import tarfile
import glob
from PIL import Image
import matplotlib.pyplot as plt
import re
import numpy as np
import random
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.layers import Activation
from keras.layers import Input, Lambda, Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
from keras.models import Sequential, Model
from keras.optimizers import RMSprop

sample_size=4000
epochs = 13

def extract(filename=""):
	if len(filename)<=0 :
		file= glob.glob("*.tar*")
		if len(file)<1:
			print("No .tar file in present working dir")
		elif len(file)>1:
			print("More than one file in present working dir")
		else:
			file=file[0]
	else:
		file= filename

	if (file.endswith("tar.gz")):
	    tar = tarfile.open(file, "r:gz")
	    tar.extractall()
	    tar.close()
	elif (file.endswith("tar")):
	    tar = tarfile.open(file, "r:")
	    tar.extractall()
	    tar.close()
	else:
		print("Only tar file supported for extraction")

def explore_images(folder, file_no):
	try:
		img= Image.open("orl_faces/"+folder+"/"+file_no+".pgm")
		img.show()
	except IOError as e:
		print("I/O error({0}): {1}".format(e.errno, e.strerror))

extract()
def pgm_to_np(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search( b"(^P5\s(?:\s*#.*[\r\n])*" b"(\d+)\s(?:\s*#.*[\r\n])*" b"(\d+)\s(?:\s*#.*[\r\n])*" b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))


#img= pgm_to_np('orl_faces/s1/1.pgm')

def data_preprocessing(sample_size):
	#read the image to numpy array
	image = pgm_to_np('orl_faces/s' + str(1) + '/' + str(1) + '.pgm', 'rw+')
	#get the new dims
	dim1 = image.shape[0]
	dim2 = image.shape[1]
	count = 0
	#initialize the numpy array with the shape of [total_sample, no_of_pairs, dim1, dim2]
	x_geuine_pair = np.zeros([sample_size, 2, 1, dim1, dim2])  # 2 is for pairs
	y_genuine = np.zeros([sample_size, 1])

	for i in range(40):
		#print("&1"+str(i))
		for j in range(int(sample_size/40)):
		    rand_img_nums = random.sample(range(1, 10), 2)           
		    # read the two images
		    img1 = pgm_to_np('orl_faces/s' + str(i+1) + '/' + str(rand_img_nums[0]) + '.pgm', 'rw+')
		    img2 = pgm_to_np('orl_faces/s' + str(i+1) + '/' + str(rand_img_nums[1]) + '.pgm', 'rw+')

		    #store the images to the initialized numpy array
		    x_geuine_pair[count, 0, 0, :, :] = img1
		    x_geuine_pair[count, 1, 0, :, :] = img2
		    
		    #as we are drawing images from the same directory we assign label as 1. (genuine pair)
		    y_genuine[count] = 1
		    count += 1
	count = 0
	x_imposite_pair = np.zeros([sample_size, 2, 1, dim1, dim2])
	y_imposite = np.zeros([sample_size, 1])
	for i in range(int(sample_size/10)):
		#print("2"+str(i))
		for j in range(10):
			#print("*2"+ str(j))
			#read images from different directory (imposite pair)
			rand_folder_nums = random.sample(range(1, 40), 2)

			img1 = pgm_to_np('orl_faces/s' + str(rand_folder_nums[0]) + '/' + str(j + 1) + '.pgm', 'rw+')
			img2 = pgm_to_np('orl_faces/s' + str(rand_folder_nums[1]) + '/' + str(j + 1) + '.pgm', 'rw+')

			x_imposite_pair[count, 0, 0, :, :] = img1
			x_imposite_pair[count, 1, 0, :, :] = img2
			#as we are drawing images from the different directory we assign label as 0. (imposite pair)
			y_imposite[count] = 0
			#print("c:"+str(count))
			count += 1
	            
	#concatenate the genuine and imposite( we will do shuffle during train test split)
	X = np.concatenate([x_geuine_pair, x_imposite_pair], axis=0)/255
	Y = np.concatenate([y_genuine, y_imposite], axis=0)

	return X, Y

X, Y = data_preprocessing(sample_size)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.25)

def build_siamese(input_shape):
    
    seq = Sequential()
    
    filters = [6, 12]
    kernel_size = 3
   
    #conv layer 1
    seq.add(Convolution2D(filters[0], kernel_size, kernel_size, input_shape=input_shape,
                          border_mode='valid', dim_ordering='th'))
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=(2, 2)))  
    seq.add(Dropout(.20))
    
    #conv layer 2
    seq.add(Convolution2D(filters[1], kernel_size, kernel_size, border_mode='valid', dim_ordering='th'))
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th')) 
    seq.add(Dropout(.20))

    #flatten 
    seq.add(Flatten())
    seq.add(Dense(128, activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(50, activation='relu'))
    return seq

input_dim = x_train.shape[2:]
img_a = Input(shape=input_dim)
img_b = Input(shape=input_dim)

siamese = build_siamese(input_dim)
feat_vecs_a = siamese(img_a)
feat_vecs_b = siamese(img_b)

def energy_func(vects):
	# we use eucledian distance
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def energy_func_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

distance = Lambda(energy_func, output_shape=energy_func_output_shape)([feat_vecs_a, feat_vecs_b])
rms = RMSprop()

model = Model(input=[img_a, img_b], output=distance)

def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

model.compile(loss=contrastive_loss, optimizer=rms)
img_1 = x_train[:, 0]
img2 = x_train[:, 1]

model.fit([img_1, img2], y_train, validation_split=.25,
          batch_size=128, verbose=2, nb_epoch=epochs)