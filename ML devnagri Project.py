# Standard useful data processing imports
import random
import numpy as np
import pandas as pd
# Visualisation imports
import matplotlib.pyplot as plt
import seaborn as sns
# Scikit learn for preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# Keras Imports - CNN
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.models import load_model



# Import the data
dataset = pd.read_csv("E:\ML project devnagri\data.csv\data.csv")
print(dataset.head())

x = dataset.values[:,:-1] / 255.0
y = dataset['character'].values

# Free memory
del dataset
n_classes = 46 # Number of classes

# Now let's visualise a few random images
img_width_cols = 32
img_height_rows = 32
cutsomcmap = sns.dark_palette("white", as_cmap=True)
random_idxs = random.sample(range(1, len(y)), 4)
plt_dims = (15, 2.5)
f, axarr = plt.subplots(1, 4, figsize=plt_dims)
it = 0
for idx in random_idxs:
    image = x[idx, :].reshape((img_width_cols, img_height_rows)) * 255
    axarr[it].set_title(y[idx].split("_")[-1])
    axarr[it].axis('off')
    sns.heatmap(data=image.astype(np.uint8), cmap=cutsomcmap, ax=axarr[it])
    it = it+1
    
# Let's split the data into train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

# Encode the categories
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
y_train = to_categorical(y_train, n_classes)
y_test = to_categorical(y_test, n_classes)

im_shape = (img_height_rows, img_width_cols, 1)
x_train = x_train.reshape(x_train.shape[0], *im_shape) # Python TIP :the * operator unpacks the tuple
x_test = x_test.reshape(x_test.shape[0], *im_shape)


#CNN MODEL BUILDING
cnn = Sequential()

kernelSize = (3, 3)
ip_activation = 'relu'#REctified Linear Unit
ip_conv_0 = Conv2D(filters=32, kernel_size=kernelSize, input_shape=im_shape, activation=ip_activation)
cnn.add(ip_conv_0)

# Add the next Convolutional+Activation layer
ip_conv_0_1 = Conv2D(filters=64, kernel_size=kernelSize, activation=ip_activation)
cnn.add(ip_conv_0_1)

# Add the Pooling layer
pool_0 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")
cnn.add(pool_0)

ip_conv_1 = Conv2D(filters=64, kernel_size=kernelSize, activation=ip_activation)
cnn.add(ip_conv_1)
ip_conv_1_1 = Conv2D(filters=64, kernel_size=kernelSize, activation=ip_activation)
cnn.add(ip_conv_1_1)

pool_1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")
cnn.add(pool_1)

# Let's deactivate around 20% of neurons randomly for training
drop_layer_0 = Dropout(0.2)
cnn.add(drop_layer_0)

flat_layer_0 = Flatten()
cnn.add(Flatten())

# Now add the Dense layers
h_dense_0 = Dense(units=128, activation=ip_activation, kernel_initializer='uniform')
cnn.add(h_dense_0)
# Let's add one more before proceeding to the output layer
h_dense_1 = Dense(units=64, activation=ip_activation, kernel_initializer='uniform')
cnn.add(h_dense_1)

op_activation = 'softmax'
output_layer = Dense(units=n_classes, activation=op_activation, kernel_initializer='uniform')
cnn.add(output_layer)

opt = 'adam'
loss = 'categorical_crossentropy'
metrics = ['accuracy']
# Compile the classifier using the configuration we want
cnn.compile(optimizer=opt, loss=loss, metrics=metrics)

history = cnn.fit(x_train, y_train,
                  batch_size=32, epochs=2,
                  validation_data=(x_test, y_test))

scores = cnn.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

#Save the Model
cnn.save('devnagri_model.h5')

#load the Model
model = load_model('devnagri_model.h5')

print(model.summary())

#Accuracy
print("Accuracy:",model.evaluate(x_test, y_test, verbose=1)[1]*100)

#Printing All the Characters
dataset = pd.read_csv("E:\ML project devnagri\data.csv\data.csv")
char_names = dataset.character.unique()  
rows =10;columns=5;
fig, ax = plt.subplots(rows,columns, figsize=(8,16))
for row in range(rows):
    for col in range(columns):
        ax[row,col].set_axis_off()
        if columns*row+col < len(char_names):
            x = dataset[dataset.character==char_names[columns*row+col]].iloc[0,:-1].values.reshape(32,32)
            x = x.astype("float64")
            x/=255
            ax[row,col].imshow(x, cmap="binary")#cmpa="binary"
            ax[row,col].set_title(char_names[columns*row+col].split("_")[-1])#after splitting print last keyword

           
plt.subplots_adjust(wspace=1, hspace=1)  

#predictions
predict = model.predict(x_test,verbose=0)

"""
#predict some random images
predict = cnn.predict(x_test,verbose=0)
predict1=np.argmax(predict, axis=1)
predict2=np.argmax(y_test,axis=1)
train_labels = os.listdir("E:\ML project devnagri\Images\Images")
"""

#functions
def _low_confidence_idx(predicted, y, confidence=0.8):
    #"""Get indexes of low confidence predictions"""
    a = []
    for i in range(predicted.shape[0]):
        if predicted[i][np.argsort(predicted[i])[-1]]<confidence:#if inequlaity changes you will get high accuracy results
            a.append(i)
    return a

def get_low_confidence_predictions(predictions, X, y, confidence=0.8):
    #"""get all info about ambiguous predictions"""
    idx = _low_confidence_idx(predictions, y, confidence=confidence)
    results = []
    for i in idx:
        result = dict()
        result["image"] = X[i].reshape(X.shape[1],X.shape[2])
        result["true_class"] = le.inverse_transform(np.argmax(y[i]))
        top2 = np.argsort(predictions[i])[-2:]
        predicted_classes = []
        for j in top2[::-1]:
            predicted_classes.append((le.inverse_transform(j), predictions[i][j]))
        result["predicted_classes"]=predicted_classes
        results.append(result)
    return results
#END of Functions


"""Display source image, true and predicted (top 2) classes with softmax score"""
low_conf = get_low_confidence_predictions(predict, x_test, y_test, confidence=0.8)

rows =10;columns=8
fig, ax = plt.subplots(rows,columns, figsize=(13,24))
for row in range(rows):
    for col in range(columns):
        ax[row,col].set_axis_off()
        s = ""
        for p in low_conf[row*columns+col]["predicted_classes"]:
            s += "{}:({:0.2f})\n".format(p[0].split("_")[-1], p[1])
        true_class = low_conf[row*columns+col]["true_class"].split("_")[-1]
        ax[row,col].set_title("True:{}\n\nPredicted:\n{}".format(true_class,s))
        g=low_conf[row*columns+col]["image"]
        g=g.astype("float64")
        ax[row,col].imshow(g, cmap="hot")

plt.subplots_adjust(wspace=2, hspace=2)   





