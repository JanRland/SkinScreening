# SkinScreening
This repository contains the preprocessing and the final convolutional neural networks for the skin lesion classification of the [HAM10000](https://www.nature.com/articles/sdata2018161) data set. The data was analysed in order to be used in the [Virtual Doctor](https://pubmed.ncbi.nlm.nih.gov/31607340/) project as a non-invasive diagnostic tool. The project was partily funded by hessen.AI.


# Data 
For training the convolutional neural networks (CNNs), we used the training and test data set provided by the [HAM10000](https://www.nature.com/articles/sdata2018161) challenge. Data augmentation and upsampling was used to stratify the data set and learning rate annihilation for the training process. From current literature, we identified CNN archtectures that are commonly used for image classification tasks and applied transfer learning. The results for the evaluation on the test set are given in the following table:

| Architecture | Accuracy | Weighted F1 | MCC |
|------------- | ------------- |------------- | ------------- | 
| DenseNet169 | 0.836 | 0.831 | 0.717 |
| DenseNet201 | 0.835 | 0.828 | 0.714 |
| Inception_V3 | 0.829 | 0.821 | 0.701 |
| ResNet50 | 0.821 | 0.815 | 0.691 |
| ResNet101 | 0.819 | 0.814 | 0.690 |
| ResNet18 | 0.817 | 0.813 | 0.686 |
| ResNet34 | 0.815 | 0.808 | 0.679 |
| DenseNet121 | 0.810 | 0.804 | 0.673 |
| VGG19 | 0.772 | 0.768 | 0.610 |
| SqueezeNet | 0.750 | 0.757 | 0.597 |
| AlexNet | 0.723 | 0.724 | 0.549 |


Three different evaluation metrices were used to analyse the final result. This avoids any manipulation by the applied augmentation/upsampling techniques. 

# Getting started
The augmented data can be generated using the Preprocessing.py script in the src folder. The training routine is defined by the Training.py script in the src folder. The weights of the resulting DenseNet169 is saved in the result folder. In order to use 


## Requirements
The python scripts were tested with following packages and versions: 

   * torch 2.1.0
   * torchvision 0.16.0
   * PIL 9.4.0
   * imblearn 0.11.0
   * sklearn 1.2.2
   * pandas 2.0.3
   * numpy 1.26.0


# Publications
Please cite following publication if you use the results:


# Authors
   * Dmitry Degtyar, dmitry.degtyar@yahoo.com, main contributor and analyst
   * Jan Benedikt Ruhland, jan.ruhland@hhu.de, supervisor
   * Prof. Dr. Dominik Heider, dominik.heider@hhu.de, principal inversitgator


# License
MIT license (see license file). 


# Acknowledgments
We want to thank the German state Hessen for the financial support of the project. Furthermore, the  Marburger Rechen-Cluster 3 (MaRC3) for providing the required computational ressources. 
