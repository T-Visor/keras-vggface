import tensorflow
from tensorflow.keras.models import  Model
from tensorflow.keras.layers import Flatten, Dense, Input
from keras_vggface.vggface import VGGFace
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#custom parameters
nb_class = 7
hidden_dim = 512

vgg_model = VGGFace(include_top=False, input_shape=(200, 200, 3))
last_layer = vgg_model.get_layer('pool5').output
x = Flatten(name='flatten')(last_layer)
x = Dense(hidden_dim, activation='relu', name='fc6')(x)
x = Dense(hidden_dim, activation='relu', name='fc7')(x)
out = Dense(nb_class, activation='softmax', name='fc8')(x)
custom_vgg_model = Model(vgg_model.input, out)

custom_vgg_model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Train your model as usual-------------------------------------------
# create data generator
datagen = ImageDataGenerator(rotation_range=30, rescale=1.0/255.0)

dataset = 'cropped/images'

# prepare iterators
train_it = datagen.flow_from_directory(dataset,
                                        class_mode='categorical', 
                                        batch_size=16, 
                                        target_size=(200, 200))

test_it = datagen.flow_from_directory(dataset,
                                      class_mode='categorical', 
                                      batch_size=16, 
                                      target_size=(200, 200))

# fit model
history = custom_vgg_model.fit_generator(train_it, steps_per_epoch=len(train_it),
                              validation_data=test_it, validation_steps=len(test_it), epochs=60,
                              verbose=1)

