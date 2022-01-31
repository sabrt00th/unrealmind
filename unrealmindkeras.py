## imports all the keras modules we need
from keras import layers,models, optimisers
from keras.preprocessing.image import ImageDataGenerator

## adds all of the model's layers
model = keras.Sequential(
    [
        layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3))
        layers.MaxPooling2D((2,2))
        layers.Conv2D(64,(3,3),activation='relu')
        layers.MaxPooling2D((2,2))
        layers.Conv2D(128,(3,3),activation='relu')
        layers.MaxPooling2D((2,2))
        layers.Conv2D(128,(3,3),activation='relu')
        layers.MaxPooling2D((2,2))
        layers.Flatten()
        layers.Dense(512,activation='relu')
        layers.Dense(1,activation='sigmoid')
    ]
)

model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc']) # configures the model for training
train_datagen,test_datagen = ImageDataGenerator(rescale=1./255) # rescales the input (0-255 range) into a 0-1 range. don't ask, I don't get it either.
base_dir = '/[insertPath]/unrealmindscreenshots' # sets the base directory. again, replace [insertPath] with your personal path.

train_dir = '/[insertPath]/unrealmindscreenshots/train' # sets the training directory.
train_generator = train_datagen.flow_from_directory(train_dir,target_size=(150,150),batch_size=20,class_mode='binary') # target_size needs to match the width+height of the images you're planninng to train this thing on

validation_dir = '/[insertPath]/unrealmindscreenshots/validation' # sets the validation directory.
validation_generator = test_datagen.flow_from_directory(validation_dir,target_size=(150,150),batch_size=20,class_mode='binary') # ditto.

test_loss,test_acc = model.evaluate_generator(test_generator,steps=50) # 50 drumrolls, please...

print('test acc:', test_acc) # prints the test accuracy. what'd you expect, a panini?
