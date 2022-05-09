import os
from turtle import color
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


'''
激活函数的作用
optimizer: 调整参数，使得损失函数趋近全局最小
'''

train_data_dir = 'D:/NotOnlyCode/DIP/downloads/train'
test_data_dir = 'D:/NotOnlyCode/DIP/downloads/test'
batch_size = 15

def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150,150,3)))
    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))
    # model.summary()

    model.compile(loss='categorical_crossentropy',  # 定义模型的loss func，optimizer，
                  optimizer=optimizers.Adam(lr=0.001),  # 使用默认的lr=0.001
                  metrics=['acc'])  # 主要优化accuracy

    return model

def load_image():
    train_datagen = ImageDataGenerator(
        rescale=1./225,
        shear_range=0.2,
        zoom_range=0.2, #随机缩放扩大
        rotation_range=30,
        horizontal_flip=True,
        vertical_flip=True,)

    #测试集的数据不做图像增强处理
    test_datagen = ImageDataGenerator(
        rescale=1./225)
    

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size = (150, 150),
        batch_size=batch_size,
        class_mode='categorical' #多分类的评估指标
        )
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size = (150, 150),
        batch_size=batch_size,
        class_mode='categorical'
        )

    return train_generator, test_generator

if __name__ == "__main__":
    print(">>>>>>>>>>>Start>>>>>>>>>>>>>")
    model = build_model()
    
    train_generator, test_generator = load_image()
    # for data_batch, lable_batch in train_generator:
    #     print('data_batch_shape:', data_batch.shape)
    #     print('label_batch_shape:', lable_batch.shape)
    #     break
    # data_batch_shape: (20, 150, 150, 3)
    # label_batch_shape: (20, 3)


    # 拟合模型
    history = model.fit_generator(
        train_generator,
        # steps_per_epoch=10,
        epochs=100,
        validation_data=test_generator,
        validation_steps=50,
        shuffle=True
    )

    acc = history.history['acc']
    val_acc = history.history['val_loss']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    
    plt.plot(epochs, loss, 'bo',label = 'Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss', color='#FF0000')
    plt.title('train and validation loss')
    plt.legend()
    plt.savefig('./loss.jpg')
    plt.show()

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc' ,color='#FF0000')
    plt.legend()
    plt.savefig('./acc.jpg')

    # model.save('wheat_leaf.h5')

    


