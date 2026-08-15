\# 🐱🐶 Cat vs Dog Image Classifier



An AI-powered web application that classifies images as \*\*Cat 🐱, Dog 🐶, or Neither\*\* using a trained TensorFlow/Keras model and a Flask web interface.



\## 📌 Project Overview



This project is an image classification application that allows users to:



\* Upload an image from their computer

\* Enter an image URL

\* Preview the image being analyzed

\* Classify the image as Cat or Dog

\* Detect images that are not cats or dogs

\* Display the prediction confidence score

\* Reset the application and analyze another image



The application uses a two-stage image analysis approach. \*\*MobileNetV2\*\*, pre-trained on ImageNet, is used as an initial validation layer to identify whether an image is likely to contain a cat or dog. The image is then passed to the custom trained TensorFlow/Keras model for the final Cat/Dog classification.



\## 🛠️ Technologies Used



\* Python 3.10

\* Flask

\* TensorFlow

\* Keras

\* MobileNetV2

\* NumPy

\* Pillow

\* Requests

\* HTML

\* CSS

\* JavaScript



\## 🧠 Machine Learning



The project uses:



\### MobileNetV2



A pre-trained MobileNetV2 model with ImageNet weights is used as an initial image validation step.



It helps filter out images containing objects such as cars, laptops, flowers, etc.



\### Custom Cat/Dog Model



A trained TensorFlow/Keras binary classification model is used to perform the final Cat vs Dog prediction.



The model accepts images resized to:



`128 × 128`



and normalized pixel values.



\## 🔄 Application Workflow



```text

User

&#x20; ↓

Upload Image / Image URL

&#x20; ↓

Image Preprocessing

&#x20; ↓

MobileNetV2

&#x20; ↓

Cat/Dog Validation

&#x20; ↓

Custom Keras Model

&#x20; ↓

Cat / Dog / Uncertain

&#x20; ↓

Display Result

```



\## 📂 Project Structure



```text

cat-dog-image-classifier/

│

├── cat\_dog\_classifier.py

├── train\_cat\_dog.py

├── cat\_dog\_model.keras

├── requirements.txt

├── README.md

├── .gitignore

│

├── training\_set/

└── test\_set/

```



> The training and test datasets may be excluded from the GitHub repository because of their size.



\## ⚙️ Installation



Clone the repository:



```bash

git clone https://github.com/bhavika-raut/cat-dog-image-classifier.git

```



Navigate to the project:



```bash

cd cat-dog-image-classifier

```



Install the required packages:



```bash

pip install -r requirements.txt

```



\## ▶️ Run the Application



Run:



```bash

python cat\_dog\_classifier.py

```



The Flask application will start locally.



Open the following address in your browser:



```text

http://127.0.0.1:5000

```



\## 🖼️ Features



\### Image Upload



Users can select an image directly from their computer.



\### Image URL



Users can provide an image URL for classification.



\### Image Preview



The image being analyzed is displayed in the application.



\### Cat/Dog Detection



The application predicts:



\* 🐱 Cat

\* 🐶 Dog

\* ❓ Uncertain

\* ❌ Neither / Not a Cat or Dog



\### Confidence Score



The prediction score is displayed with the classification result.



\## 🎯 Future Improvements



\* Improve model accuracy with a larger and more diverse dataset

\* Add confusion matrix and model evaluation metrics

\* Add prediction history

\* Deploy the application online

\* Improve UI/UX

\* Add support for additional animal categories



\## 👩‍💻 Author



\*\*Bhavika Raut\*\*



GitHub:

https://github.com/bhavika-raut



```

```



