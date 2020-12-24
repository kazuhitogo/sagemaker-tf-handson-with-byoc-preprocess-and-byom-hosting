import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
import argparse, os, numpy as np, json
from glob import glob

def model(x_train, y_train, x_test, y_test, epochs, dropout_rate):
    model = Sequential()
    model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(28,28,1),padding="same"))
    model.add(PReLU())
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=32,kernel_size=(3,3),padding="same"))
    model.add(PReLU())
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2),padding="same")) # 28,28 -> 14,14
    model.add(Conv2D(filters=32,kernel_size=(3,3),padding="same"))
    model.add(PReLU())
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=32,kernel_size=(3,3),padding="same"))
    model.add(PReLU())
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2),padding="same")) # 14,14 -> 7,7
    model.add(Conv2D(filters=32,kernel_size=(3,3),padding="same"))
    model.add(PReLU())
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=32,kernel_size=(3,3),padding="same"))
    model.add(PReLU())
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2),padding="same")) # 7,7 -> 4,4
    model.add(Conv2D(filters=32,kernel_size=(3,3),padding="same"))
    model.add(PReLU())
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=32,kernel_size=(3,3),padding="same"))
    model.add(PReLU())
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2),padding="same")) # 4,4 -> 2,2
    model.add(Conv2D(filters=32,kernel_size=(2,2),padding="same"))
    model.add(PReLU())
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2),padding="same")) # 2,2 -> 1,1
    model.add(Flatten())
    model.add(Dense(10,activation="softmax"))

    model.compile(optimizer=Adam(lr=0.0001),metrics=['accuracy'],loss="categorical_crossentropy")
    model.fit(x_train, y_train,batch_size=16,epochs=epochs,validation_data=(x_test,y_test))

    return model


def _load_training_data(base_dir):
    """Load MNIST training data"""
    x_train = np.load(os.path.join(base_dir, 'train_x.npy'))
    y_train = np.load(os.path.join(base_dir, 'train_y.npy'))
    return x_train, y_train


def _load_testing_data(base_dir):
    """Load MNIST testing data"""
    x_test = np.load(os.path.join(base_dir, 'test_x.npy'))
    y_test = np.load(os.path.join(base_dir, 'test_y.npy'))
    return x_test, y_test


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--dropout-rate', type=float, default=0.5)

    return parser.parse_known_args()


if __name__ == "__main__":
    args, unknown = _parse_args()
    train_data, train_labels = _load_training_data(args.train)
    eval_data, eval_labels = _load_testing_data(args.test)

    mnist_classifier = model(train_data, train_labels, eval_data, eval_labels,args.epochs,args.dropout_rate)
    save_model_path = os.path.join(args.sm_model_dir, '000000001')
    mnist_classifier.save(save_model_path)
