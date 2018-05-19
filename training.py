from data import Data
import keras





#params
classes = ['fish','humain','laptop','backpack','dog','billiard']
data = Data(classes,0.9)



#loading pretrained inception_resnet model
inception_resnet = keras.applications.InceptionResNetV2(include_top=False,weights='imagenet',input_shape=(139,139,3))



x = inception_resnet.output

x = keras.layers.GlobalAveragePooling2D()(x)

x = keras.layers.Dense(1024,activation='relu')(x)


predictions = keras.layers.Dense(6,activation='softmax')(x)







model = keras.models.Model(inputs=inception_resnet.input,outputs=predictions)




for layer in inception_resnet.layers :
    layer.trainable = False





images_train, labels_train, images_test, labels_test = data.getAllDataAtOnes()


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(x=images_train,y=labels_train,batch_size=100,epochs=10,shuffle=True,initial_epoch=0, validation_split=0.2)



scores = model.evaluate(x=images_test,y=labels_test)


print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))



model.save("model.h5")











