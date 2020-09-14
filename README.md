# Pneumonia Classifier
A ConvNet trained on a [Kaggle Dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) that takes chest X-ray as an input and predicts whether the patient has pneumonia or not. Integrated it into a web app with Flask and deployed it on Heroku.

## Details
Achieved 0.83 recall, 0.91 precision and 0.87 f1-score. The model used to overfit a lot, so used data augmentation, batchnorm and dropouts to reduce variance. Used this [GitHub repo](https://github.com/mtobeiyf/keras-flask-deploy-webapp) as a reference and modified it to suit the model and the website.
<br>Link to the website: https://pnclassify.herokuapp.com/

## Tech Stack
- DL Model : Python, TensorFlow, Cv2
- Backend : Flask
- Frontent : HTML, CSS, Bootsrap, JavaScript
- Deployment : Heroku
