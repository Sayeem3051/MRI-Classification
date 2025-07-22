# Brain Tumor MRI Classification

This is a Streamlit web application that uses two deep learning models (a custom CNN and MobileNetV2) to classify brain tumors from MRI images into one of four categories: glioma, meningioma, no tumor, or pituitary tumor.

## Features

- **Upload MRI Images**: Users can upload an MRI image in JPG, JPEG, or PNG format.
- **Dual Model Prediction**: Choose between a custom-built CNN and a fine-tuned MobileNetV2 for classification.
- **Instant Results**: View the predicted tumor type along with the model's confidence score.
- **Detailed Probabilities**: See the probability for each potential tumor class.

## How to Run Locally

1.  **Clone the repository:**
    ```sh
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```sh
    streamlit run mri/brain_tumor_app.py
    ```

## Models Used

- **Custom CNN**: A convolutional neural network built from scratch and trained on the brain tumor MRI dataset.
- **MobileNetV2**: A pre-trained model, fine-tuned on the same dataset for high-accuracy transfer learning. 
