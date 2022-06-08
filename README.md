# Solving-Image-Classification-Problems-with-Convolutional-Neural-Network

The dataset used in this project is the Chest X-ray dataset: https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia

This project consists of two goals: (1) compare the performance of multiple CNN pre-trained models  and (2) propose a new model that can improve the performance of the best pre-trained model. For the first step, selected two of the pre-trained models and compared their performance with the current dataset without changing any aspect of these networks:
1. AlexNet

2.ResNet

however the essential steps your code should include are:

• Data Cleaning and analysis

• Loading and adjusting the pre-trained models

• Training and testing different models

• Plot the results comparing different networks

As it can be seen in the figure1 that the model trains with the accuracy of 61% for the Test Data and for the training data the accuracy is as high as 93%.

![image](https://user-images.githubusercontent.com/99655823/172712231-99249b14-64b9-4a68-a205-fc4c563fb559.png)

In the figure 2 it can be clearly seen that earlier the loss was very high, as high as 40% which came down further as the model re-tuned itself and is mainly quantifies on the output based on the set of inputs, which is referred to as parameter values. I used “categorical_crossentropy” as it is best for multiclass classification model where there are two or more output labels.

![image](https://user-images.githubusercontent.com/99655823/172712326-b2526f9f-a079-40ac-b6e1-fd46b7916fb5.png)

Finally, after training and testing the model, we can see in below figure that puts the light on the confusion matrix for the AlexNet and explains the division of images based on the training of the model.

![image](https://user-images.githubusercontent.com/99655823/172712440-327c4fb0-0a51-4231-babe-779dcc261163.png)

Finally, after training and testing the model, we can see in below figure that puts the light on the confusion matrix for the ResNet and explains the division of images based on the training of the model.

![image](https://user-images.githubusercontent.com/99655823/172712672-a9a29532-2c5e-42a6-89dc-e7a70b5fbb93.png)

Finally, after training and testing the model, we can see in below figure that puts the light on the confusion matrix for the ResNet and explains the division of images based on the training of the model.

![image](https://user-images.githubusercontent.com/99655823/172712770-ee7f39ea-d5bd-4fcb-b242-a94d23115a02.png)



As it is seen that ResNet finally gave us the better results in learning and understanding the model to train itself and find the best result among the rest, the table below clearly brings light to the same.

![image](https://user-images.githubusercontent.com/99655823/172712020-67123419-504b-480d-a6e5-cd13b9dc6481.png)


I believe ResNet performed better than the AlexNet because of reasons mentioned below:
1. Because of the framework of ResNet as it was made to train ultra-deep neural networks and by that, I mean its network can contain hundreds or thousands of layers and still achieve great performance
2. ResNet has almost 5 times training time for the model which makes it better in terms of learning and prediction of the data.
