import cv2
import lxml.etree as et
import os
import time
import numpy as np

class Data :


    def __init__(self,classes,training_rate):

        # path to data
        self.train_data = "./data/training/"

        # path to annotation
        self.train_annotation = "./data/annotations/training"




        #params
        self.batch_size = 30
        self.classes = classes
        self.index = 0
        self.training_rate = training_rate

        self.images_train = []
        self.images_test = []
        self.labels_train = []
        self.labels_test = []

        self.getData()


    def getAllDataAtOnes(self) :

        images = []
        labels = []

        imageDirectories_train = [
            d for d in os.listdir(self.train_data)
            if os.path.isdir(os.path.join(self.train_data, d))
        ]

        # extract images
        for d in imageDirectories_train:

            for sd in os.listdir(os.path.join(self.train_data, d)):

                for img in os.listdir(os.path.join(self.train_data, d, sd)):
                    index_class = self.classes.index(d)
                    label = np.zeros(len(self.classes))
                    label[index_class] = 1.0
                    labels.append(label)

                    annotationFile = os.path.join(self.train_annotation, d, sd, img.split('.')[0] + ".xml")
                    xmin, ymin, xmax, ymax = self.getLocation(annotationFile)

                    image = cv2.imread(os.path.join(self.train_data, d, sd, img))
                    image = self.getImageByRegion(image, xmin, ymin, xmax, ymax)
                    image = cv2.resize(image, (139, 139), 0, 0, cv2.INTER_LINEAR)
                    image = np.multiply(image, 1.0 / 255.0)

                    images.append(image)

        images = np.array(images)
        labels = np.array(labels)



        training_nbr = int(len(images) * self.training_rate)
        self.images_train = images[:training_nbr]
        self.labels_train = labels[:training_nbr]
        self.images_test = images[training_nbr + 1:]
        self.labels_test = labels[training_nbr + 1:]


        return self.images_train,self.labels_train,self.images_test,self.labels_test







    def getData(self):

        images = []
        labels = []



        imageDirectories_train = [
            d for d in os.listdir(self.train_data)
            if os.path.isdir(os.path.join(self.train_data,d))
        ]



        #extract images
        for d in imageDirectories_train :

            for sd in os.listdir(os.path.join(self.train_data,d)) :

                for img in os.listdir(os.path.join(self.train_data,d,sd)) :

                    index_class = self.classes.index(d)
                    label = np.zeros(len(self.classes))
                    label[index_class] = 1.0
                    labels.append(label)




                    annotationFile = os.path.join(self.train_annotation,d,sd,img.split('.')[0]+".xml")
                    xmin,ymin,xmax,ymax = self.getLocation(annotationFile)

                    image = cv2.imread(os.path.join(self.train_data,d,sd,img))
                    image = self.getImageByRegion(image,xmin,ymin,xmax,ymax)
                    image = cv2.resize(image,(229,229),0,0,cv2.INTER_LINEAR)
                    image = np.multiply(image,1.0/255.0)


                    images.append(image)




        images = np.array(images)
        labels = np.array(labels)


        #divide training and test data
        perm = np.arange(len(images))
        np.random.shuffle(perm)
        images = images[perm]
        labels = labels[perm]



        training_nbr = int(len(images)*self.training_rate)
        self.images_train = images[:training_nbr]
        self.labels_train = labels[:training_nbr]
        self.images_test = images[training_nbr+1:]
        self.labels_test = labels[training_nbr+1:]








    def getNextBatch(self):



        endBatch = self.index + self.batch_size


        images = self.images_train[self.index:endBatch]
        labels = self.labels_train[self.index:endBatch]

        if self.index >= len(self.images_train) :
            self.index = 0
        else  :
            self.index += self.batch_size


        return images,labels




    def getImageByRegion(self,image,xmin,ymin,xmax,ymax) :
        return image[ymin:ymax,xmin:xmax]

    def getLocation(self,path) :

        f = open(path,'r')
        str = f.read()
        tree = et.fromstring(str)

        xmin = 0
        for p in tree.xpath('//xmin') :
            xmin = p.text

        ymin = 0
        for p in tree.xpath('//ymin') :
            ymin = p.text

        xmax = 0
        for p in tree.xpath('//xmax') :
            xmax = p.text

        ymax = 0
        for p  in tree.xpath('//ymax') :
            ymax = p.text

        return int(xmin),int(ymin),int(xmax),int(ymax)

    def showImgByPath(self,path) :
        im = cv2.imread(path)
        cv2.imshow("aaa", im)
        cv2.waitKeyEx()

    def showImg(self,image) :
        cv2.imshow("img",image)
        cv2.waitKeyEx()


















