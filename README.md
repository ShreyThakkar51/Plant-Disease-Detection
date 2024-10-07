
# Plant Disease Detection

This project implements a deep learning-based solution for detecting plant diseases from images. The model is trained using a Convolutional Neural Network (CNN) and deployed via an interactive user interface using Gradio.


## Features

- Image Classification: Detects various plant diseases from images.
- Deep Learning Model: A CNN trained with Keras and TensorFlow to classify plant diseases.
- Interactive UI: Powered by Gradio, allowing easy interaction with the model.
- Preprocessing: Handles image preprocessing like resizing and normalization before feeding into the model.
- Model Persistence: The trained model is saved as best_model.h5 and can be reused for future predictions.

## Project Structure

- Plant_disease_detection.ipynb: Jupyter notebook containing the code for training the model and setting up the Gradio interface.
- best_model.h5: The saved Keras model after training.

## Requirements

- Python 3.x
- TensorFlow/Keras
- Gradio
- Numpy
- Matplotlib


## Installation

Install the required libraries by running:

```bash
  pip install tensorflow keras gradio numpy matplotlib
```
 ## Usage

- Model Training: Run the notebook Plant_disease_detection.ipynb to train the model. After training, the model is saved as best_model.h5.

- Prediction: Load the saved model and use the Gradio interface for making predictions on new images.
   
## Example usage:

```python
from keras.models import load_model
model = load_model('best_model.h5')

# Function to predict plant disease
def prediction(img):
    # Preprocessing and prediction logic here
    pass
```
- Deploy: Run the Gradio interface for user interaction:

```python
gr.Interface(fn=prediction, inputs=gr.inputs.Image(shape=(256,256)), outputs=gr.outputs.Label()).launch()
```

## Model Details

- Input image size: (256, 256)
- The model is a CNN with several convolutional layers, trained to classify plant diseases based on input images.
