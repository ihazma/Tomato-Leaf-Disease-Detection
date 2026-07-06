# 🍅 Tomato Leaf Disease Detection

A deep learning-based web application for detecting tomato leaf diseases from images using **EfficientNetB0** with **Transfer Learning** and **Fine-Tuning**. The application is built with **TensorFlow** and deployed locally using **Streamlit**.

---

## Features

- Detects **10 tomato leaf classes**
- Transfer Learning with EfficientNetB0
- Fine-Tuning for improved performance
- Interactive Streamlit web interface
- Displays prediction confidence
- Shows Top-3 predicted classes
- Works with uploaded leaf images

---

## Classes

- Bacterial Spot
- Early Blight
- Late Blight
- Leaf Mold
- Septoria Leaf Spot
- Spider Mites
- Target Spot
- Tomato Yellow Leaf Curl Virus
- Tomato Mosaic Virus
- Healthy Leaf

---

## Model Architecture

- **Base Model:** EfficientNetB0 (ImageNet pretrained)
- **Transfer Learning**
- **Data Augmentation**
- Global Average Pooling
- Dense (256 ReLU)
- Dropout (0.3)
- Softmax Output Layer

---

## Dataset

This project uses the **PlantVillage Tomato Leaf Dataset**.

The dataset contains images of healthy and diseased tomato leaves across 10 classes.

---

## Technologies Used

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Pillow

---

## Project Structure

```
Tomato-Leaf-Disease-Detection/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── Tomato_Leaf_Disease_Detection.ipynb
├── models/
│   └── best_model.keras
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Tomato-Leaf-Disease-Detection.git
```

Move into the project directory:

```bash
cd Tomato-Leaf-Disease-Detection
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Run the Streamlit App

```bash
streamlit run app.py
```

or

```bash
python -m streamlit run app.py
```

---

## Model Performance

- Validation Accuracy: **~95%**
- Transfer Learning using EfficientNetB0
- Fine-Tuned on the final layers for improved classification performance

---

## Future Improvements

- Improve performance on real-world field images
- Expand support for additional crop diseases
- Deploy the application online
- Add disease treatment recommendations

---

## References

- PlantVillage Dataset
- TensorFlow Documentation
- Keras Documentation
- Streamlit Documentation
- EfficientNet Research Paper

---

## Author

**Hamza Khan**

This project was developed for learning, research, and portfolio purposes.
