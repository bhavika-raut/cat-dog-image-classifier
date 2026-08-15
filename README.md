# 🐱🐶 Cat vs Dog Image Classifier

## 📌 Overview

An **AI-powered image classification web application** that identifies whether an image contains a **Cat 🐱, Dog 🐶, or Neither ❌**.

The application combines a **custom TensorFlow/Keras binary classification model** with a **pre-trained MobileNetV2 model** to provide more reliable image validation.

### 🔹 Two-Stage Classification

1️⃣ **MobileNetV2 + ImageNet** – Validates whether the uploaded image is likely to contain a cat or dog.

2️⃣ **Custom Keras Model** – Performs the final Cat vs Dog classification and provides a prediction score.

---

## 🎯 Purpose

* Build a practical **computer vision** application.
* Apply **deep learning** to real-world image classification.
* Provide an easy-to-use web interface for image prediction.
* Reduce incorrect predictions for non-cat/dog images using an additional validation model.

---

## 🚀 Features

✅ **Cat vs Dog Classification**
Classifies images as Cat 🐱 or Dog 🐶.

✅ **Non-Cat/Dog Detection**
Uses MobileNetV2 to identify images that are not cats or dogs.

✅ **Image Upload**
Upload an image directly from your computer.

✅ **Image URL Support**
Analyze an image using its URL.

✅ **Image Preview**
Displays the image being analyzed.

✅ **Prediction Score**
Shows the model's prediction score with the result.

✅ **Reset Functionality**
Reset the application and analyze another image.

✅ **Web-Based Interface**
Simple and user-friendly Flask web application.

---

## 🧠 Machine Learning Approach

The application follows a two-stage image classification pipeline:

```text
                 📷 Input Image
                       │
                       ▼
              ┌─────────────────┐
              │   MobileNetV2   │
              │    ImageNet     │
              └────────┬────────┘
                       │
              Cat / Dog detected?
                  ╱           ╲
                No             Yes
                │               │
                ▼               ▼
          ❌ Reject       Custom Keras Model
                                  │
                                  ▼
                         ┌─────────────────┐
                         │  Cat / Dog /    │
                         │    Uncertain    │
                         └─────────────────┘
```

### 🔹 MobileNetV2

A pre-trained **MobileNetV2** model with ImageNet weights is used as an initial validation layer.

It helps identify whether the input image is likely to contain a cat or dog before the custom model performs the final classification.

### 🔹 Custom Keras Model

The trained binary classification model performs the final Cat vs Dog prediction.

Input size:

```text
128 × 128 pixels
```

Pixel values are normalized to:

```text
0 – 1
```

---

## 🛠️ Technologies Used

| Technology      | Purpose                               |
| --------------- | ------------------------------------- |
| 🐍 Python       | Programming language                  |
| 🌐 Flask        | Web framework                         |
| 🧠 TensorFlow   | Deep learning framework               |
| ⚙️ Keras        | Model training and prediction         |
| 👁️ MobileNetV2 | Pre-trained image classifier          |
| 🔢 NumPy        | Numerical/image processing            |
| 🖼️ Pillow      | Image processing                      |
| 🌍 Requests     | Image URL handling                    |
| 🎨 HTML/CSS     | Web interface                         |
| ⚡ JavaScript    | Image preview and reset functionality |

---

## 📂 Project Structure

```text
cat-dog-image-classifier/
│
├── 📄 cat_dog_classifier.py
│       └── Flask web application
│
├── 📄 train_cat_dog.py
│       └── Model training script
│
├── 📄 cat_dog_model.keras
│       └── Trained TensorFlow/Keras model
│
├── 📄 requirements.txt
│       └── Python dependencies
│
├── 📄 README.md
│       └── Project documentation
│
├── 📄 .gitignore
│
├── 📁 training_set/
│       └── Training images
│
└── 📁 test_set/
        └── Testing images
```

> Dataset folders can be excluded from GitHub when they are too large.

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/bhavika-raut/cat-dog-image-classifier.git
```

### 2️⃣ Navigate to the Project

```bash
cd cat-dog-image-classifier
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Flask server:

```bash
python cat_dog_classifier.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

---

## 🖼️ Application

The web application provides:

```text
┌──────────────────────────────────────┐
│       🐾 Cat vs Dog Classifier       │
│                                      │
│  1. Upload Image                    │
│  [ Choose File ]                    │
│                                      │
│               OR                     │
│                                      │
│  2. Image URL                       │
│  [ https://... ]                    │
│                                      │
│  ┌──────────────────────────────┐   │
│  │      Image Preview           │   │
│  └──────────────────────────────┘   │
│                                      │
│       [ Predict → ]  [ ↻ ]          │
│                                      │
│  RESULT                              │
│  🐶 Dog (0.9998)                    │
└──────────────────────────────────────┘
```

---

## 🔍 Example Predictions

| Input           | Result      |
| --------------- | ----------- |
| 🐱 Cat image    | Cat 🐱      |
| 🐶 Dog image    | Dog 🐶      |
| 🚗 Car image    | Neither ❌   |
| 🌸 Flower image | Neither ❌   |
| Unclear image   | Uncertain ❓ |

---
## 🖼️ Screenshots

### 🐾 Cat vs Dog Classifier

The web application allows users to upload an image or provide an image URL and displays the analyzed image along with the prediction result.

![Cat vs Dog Classifier](screenshots/cat-dog-classifier.png)

---
## 📈 Model Details

### Input

```text
128 × 128 × 3
```

### Output

Binary classification score.

The application interprets the score as:

```text
Score > 0.50       → 🐶 Dog
Score < 0.50       → 🐱 Cat
0.40 – 0.60        → ❓ Uncertain
```

---

## 🔮 Future Improvements

🚀 Improve model accuracy with a larger and more diverse dataset.

📊 Add model evaluation metrics such as:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

📈 Add prediction history.

☁️ Deploy the application using a cloud platform.

🎨 Improve the user interface and user experience.

🐾 Extend the model to support additional animal categories.

---

## 👩‍💻 Author

### **Bhavika Raut**

AI & Machine Learning Enthusiast

🔗 GitHub:
https://github.com/bhavika-raut

---

## ⭐ If you find this project useful

Feel free to **star ⭐ the repository** and explore the project!
