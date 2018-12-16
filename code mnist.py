
# In[1]:


import numpy as np


# In[2]:


import matplotlib.pyplot as plt


# In[3]:


import tensorflow 
from tensorflow.python.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import random


# In[4]:


np.random.seed(0)


# In[5]:


(x_train, y_trian), (x_test, y_test) = mnist.load_data()


# In[6]:


print(x_train.shape)
print(x_test.shape)
print(y_trian.shape)
print(y_test.shape)


# In[7]:


assert(x_train.shape[0]==y_trian.shape[0]), "the number of the image is not equal to the number of labels"
assert(x_test.shape[0]==y_test.shape[0]), "the number of the image is not equal to the number of labels"
assert(x_train.shape[1:]==(28, 28)), "the dimensions of the images are not 28*28"
assert(x_test.shape[1:]==(28, 28)), "the dimensions of the images are not 28*28"


# In[8]:


num_de_examples = []
cols = 5
num_classes = 10
fig, axis = plt.subplots(nrows=num_classes, ncols=cols, figsize=(10, 10))
fig.tight_layout()
for i in range(cols):
    for j in range(num_classes):
        x_select = x_train[y_trian==j]
        axis[j][i].imshow(x_select[random.randint(0, len(x_select)-1), :, :], cmap=plt.get_cmap("gray"))
        axis[j][i].axis("off")
        if (i==2):
            axis[j][i].set_title(str(j))
            num_de_examples.append(len(x_select))


# In[9]:


print(num_de_examples)
plt.figure(figsize=(12,4))
plt.bar(range(0, num_classes), num_de_examples)
plt.title("distrubtion of the training datasets")
plt.xlabel("nombre de classe")
plt.ylabel("nombre des images")


# In[10]:


y_trian = to_categorical(y_trian, 10)
y_test = to_categorical(y_test, 10)


# In[11]:


x_train = x_train/255
x_test = x_test/255


# In[12]:


number_pixels = 784
x_train = x_train.reshape(x_train.shape[0], number_pixels)
x_test = x_test.reshape(x_test.shape[0], number_pixels)
print (x_train.shape)
print (x_test.shape)


# In[13]:


def create_model():
    model=Sequential()
    model.add(Dense(10, input_dim=number_pixels, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# In[14]:


model =create_model()
print(model.summary())


# In[15]:


H=model.fit(x=x_train, y=y_trian, verbose=1, validation_split=0.1,epochs=10, batch_size=200, shuffle=1)


# In[16]:


plt.plot(H.history['loss'])
plt.plot(H.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.title('loss')
plt.xlabel('epoch')


# In[17]:


plt.plot(H.history['acc'])
plt.plot(H.history['val_acc'])
plt.legend(['acc', 'val_acc'])
plt.title('accuracy')
plt.xlabel('epoch')


# In[18]:


score =model.evaluate(x_test, y_test, verbose=0)
print('test score', score[0])
print('test accuracy', score[1])

