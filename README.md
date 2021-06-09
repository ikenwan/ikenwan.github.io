## Bird Species Classifier

**Group members:** Ikechi Nwankwo (ikenwan)

### Problem we tackled:

I made a neural network classifier that is able to somewhat accuaretly guess a bird's species. 

### Algorithms/Techniques we used:

Convolutional Neural Network, Transfer Learning, Resnet18 Neural Network, Data Augmentation

### Dataset:

[Bird Dataset](https://www.kaggle.com/c/birds21sp/data)

### Itroduction:
While the goal of my project was to make the best classifier I could, I still chose to train some models that I figured wouldn't be as good just to see how they would compare. I played around with Resnet18 and Resnet34, two prebuilt models, and one non prebuilt darknet model I slightly altered. 

### Problem setup:
 I used colab to train my models, so while the code I included in this repo does have a train function, it probably won't work properly locally unless the get_data() functions are altered. Instead of having the code train, running main.py will load my already trained models and display their loss.

### My Approach:
I started by implementing a darknet model that is based significantly on the tutorial model. The only difference is that it takes 128 pixel images instead of 64 pixel images. I also made sure to use data augmentation to provide more robustness. I then trained this model from scratch only using the kaggle data. As expected this model has the highest loss.

Next I decided to add in transfer learning using a previous checkpoint. I still used data augmentation and as a result of both of these alterations, the loss was slightly reduced after I trained again on the kaggle data.

Next, I decided to use a prebuilt Resnet18 model and then train it on the kaggle data once again using data augmentations such as random cropping and resizing. This lead to another big decrease in loss.

Finally, I decided to use a Resney34 model and train that on the given data. However, without realizing I just accidentally retrained the resnet18 model again. Suprisingly, this lead to the biggest  decrease as shown below. This was probably due to the different constraints I used.

### Analysis:
[darknet loss graph](https://drive.google.com/file/d/1nsXlhKfeRlFd5g9irKICCHeZvcHZ3Uxy/view?usp=sharing)
[resnet18 loss graph](https://drive.google.com/file/d/1JGh36zT_Ga2GcmaajoAjS7K8qkc2ROOr/view?usp=sharing)



## [Project Presentation Video](https://www.youtube.com/watch?v=s99BOFDfnV0)

## Discussion:
### What problems did you encounter?


### Are there next steps you would take if you kept working on the project?


### How does your approach differ from others? Was that beneficial?

