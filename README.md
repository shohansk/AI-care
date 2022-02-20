# skin-cancer-classifier
    Web based prototype skin cancer classification using CNN models that classifiying 7 class of skin cancer based on lesions skin condition.

This repository contains all the models I have also created flask web based UI for the classification prediction.

# Details about the files
    best_model_cutix.ipynb  :  iPythonNotebook File which classification of skin cancer based on the CNN model architecture that I made myself. 

    best_model_cutix.h5  : Weights are then saved to this file for directly used for UI purpose.

    app.py  : Flask based UI file which helps in prediction of the image by running the 'best_model_cutix.h5' file in the backend for making prediction by getting image by the user and predict the output.

    index.html  : For basic Interface.

    main.css  : For styling the web interface

    main.js  : To make websites more dynamic and interactive and to make file uploads more interactive on the web

    uploads  : It contains the test image 


# Dataset 
    Link to download datasets : https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000 , https://www.kaggle.com/discdiver/mnist1000-with-one-image-folder (contains all images in 1 folder only)

# Required Libraries
    Web framework : Flask 
    Tensorflow
    Matplotlib
    Keras
    Numpy
    Pandas
    Scikit-learn

# Steps to do this project web based skin cancer clssification using Visual Studio Code
     Step 1 : Run the ‘best_model_cutix.ipynb’ file in either Google Coaboratory/Jupyter/Visual Studio Code
     Step 2 : At the final step of Training the model , save that model in the same folder in  which  the ‘app.py’ file is present.
     Step 3 : Set the path of saved Model in app.py 
     Step 4 : Now run ‘app.py’ file to get the UI of Model Prediction (Classification)
     Step 5 : Follow the localhost link to open the User Interface in Web Browser and start to classifying skin lesion conditions to know what types of skin cancer from the images uploaded by users.
