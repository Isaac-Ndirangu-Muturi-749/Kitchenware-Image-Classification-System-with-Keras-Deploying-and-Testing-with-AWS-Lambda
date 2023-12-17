import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import Xception
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler

# Path to the CSV file containing image labels
csv_path = '/kaggle/input/kitchenware-classification/train.csv'

# Read the CSV file into a DataFrame
df_train_full = pd.read_csv(csv_path, dtype={'Id': str})

# Assuming 'Id' is a column in your CSV file, and you have a directory 'images'
df_train_full['filename'] = '/kaggle/input/kitchenware-classification/images/' + df_train_full['Id'] + '.jpg'

# Split the DataFrame into training and validation sets
train_df, val_df = train_test_split(df_train_full, test_size=0.2, random_state=42)
print("train_df shape:", train_df.shape)
print("validation_df shape:", val_df.shape)

# Define image size and batch size
img_size = (224, 224)  # based on your model requirements
batch_size = 32

# Create ImageDataGenerators for training and validation
# With data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.4,
    zoom_range=0.4,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# Flow images from directories and apply data augmentation
train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='filename',
    y_col='label',          # 'label' is the target variable
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',  # based on your model requirements
    shuffle=True
)

val_datagen = ImageDataGenerator(rescale=1./255)  # Only rescale for validation

val_generator = val_datagen.flow_from_dataframe(
    val_df,
    x_col='filename',
    y_col='label',          # 'label' is the target variable
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',  # Adjust based on your model requirements
    shuffle=False  # No need to shuffle for validation
)

# Define the base model with pre-trained weights
base_model = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Create a Sequential model and add the Xception base model
model = Sequential()
model.add(base_model)

# Add additional layers on top of the Xception base model
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))  # Add dropout for regularization

# Add more dense layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3)) 

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))  

# Output layer
model.add(Dense(6))  # Adjust the number of output classes 

# Compile the model
learning_rate = 0.001
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
loss = keras.losses.CategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Define callbacks
checkpoint = ModelCheckpoint("best_model_xception.h5", monitor="val_accuracy", save_best_only=True)
early_stopping = EarlyStopping(monitor="val_loss", patience=5)

def lr_schedule(epoch):
    lr = 0.001
    if epoch > 10:
        lr *= 0.1  # Adjust timing of learning rate adjustment
    return lr
lr_scheduler = LearningRateScheduler(lr_schedule)

epochs = 20  # Increase the number of epochs for better convergence

# Fit the model with callbacks
history_xception = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    callbacks=[checkpoint, early_stopping, lr_scheduler]
)

# Save the model in the native Keras format
model.save('best_model_xception.keras')
