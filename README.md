**Classification of Pulmonary Tuberculosis from Chest X-rays**

A deep learning system that classifies chest X-ray images as Normal or Tuberculosis using a hybrid model: a ResNet50 convolutional backbone followed by a Self-attention mechanism. The repository contains the training and evaluation notebook and a small Flask web app for uploading an X-ray and getting a prediction.

**Disclaimer: This is an academic/portfolio project. It is not a medical device and must not be used for clinical diagnosis.**

Dataset
Source: T. Rahman et al., "Reliable Tuberculosis Detection Using Chest X-Ray With Deep Learning, Segmentation and Visualization," IEEE Access, vol. 8, pp. 191586-191601, 2020. DOI: 10.1109/ACCESS.2020.3031384. The authors compiled the images from several public databases.
Size used here: 4,200 images (3,500 Normal, 700 Tuberculosis).
Split: 70% train / 20% validation / 10% test, random with seed 1337 (split-folders).
Split	Normal	Tuberculosis	Total
Train	2,800	560	3,360
Validation	700	141	841
Test	350	71	421

**This project only uses the dataset from the paper. It does not reproduce the paper's methods, so the results below are not a like-for-like comparison with the paper.**

Model
Component	Details
Input	512 x 512 RGB, scaled to [0, 1]
Backbone	ResNet50 pretrained on ImageNet (output 16 x 16 x 2048, reshaped to 256 tokens)
Projection	Dense layer to 256 dimensions
Transformer encoder	2 blocks, 4 attention heads, key dimension 64, feed-forward size 512, dropout 0.1
Head	Global average pooling, Dense(128, ReLU), Dropout(0.5), Dense(1, sigmoid)
Parameters	about 25.2 million in total

**Training**

Augmentation (training only): rotation up to 5 degrees, nearest fill.
Class weights: Normal 1.0, Tuberculosis 3.0, to handle the 5:1 class imbalance.
Optimizer: RMSprop, batch size 16, binary cross-entropy.
Stage 1: ResNet50 frozen, learning rate 1e-4.
Stage 2: last 30 ResNet50 layers unfrozen, learning rate 1e-6, up to 50 epochs.
Callbacks: early stopping (patience 5), learning-rate reduction on plateau, best-model checkpoint.
Decision threshold: 0.569 (used for the confusion matrix, classification report, and the web app).
Results

Evaluated on the held-out test set (421 images, never used for training or validation).

Metric	Value
Accuracy	99.52%
Precision (TB)	100.00%
Recall / sensitivity (TB)	97.18%
F1-score (TB)	98.57%
ROC-AUC	0.9998
PR-AUC	0.9991

Confusion matrix (threshold 0.569):

	Predicted Normal	Predicted Tuberculosis
Actual Normal	350	0
Actual Tuberculosis	2	69
Repository contents
File	Description
baseline.ipynb	Data split, preprocessing, model definition, two-stage training, and evaluation
app.py	Flask app that loads the trained model and predicts on an uploaded X-ray
How to run

1. Install dependencies

bash
pip install tensorflow flask pillow numpy scipy scikit-learn pandas matplotlib seaborn opencv-python split-folders

app.py sets TF_USE_LEGACY_KERAS=1. With TensorFlow 2.16 or newer, also run pip install tf-keras.

2. Train the model

Download the dataset from Kaggle and unzip it.
Open baseline.ipynb and change the dataset paths in the first cells to your own folders.
Run all cells. The notebook saves the trained model as final_model.keras.

A GPU is recommended: training used 512 x 512 images and took roughly 5 minutes per epoch.

3. Start the web app

Put final_model.keras in the same folder as app.py, and make sure the templates/index.html page used by the app is present. Then run:

bash
python app.py

Open http://127.0.0.1:5000, upload a chest X-ray image (up to 16 MB), and the app shows the predicted class, the confidence, and the probability for each class.

Limitations
The train, validation, and test sets come from a random image-level split of one dataset. The dataset has no patient identifiers, so images from the same patient or source could appear in more than one split.
The model has not been tested on an external dataset or on images from a different hospital or X-ray machine, so real-world performance is likely lower than the scores above.
The test set has only 71 Tuberculosis cases, so each missed case changes recall noticeably.
The images are not lung-segmented. The model may rely on image artifacts rather than only on the lungs, which is why explainability (for example Grad-CAM) is a planned next step.
Possible next steps
Add Grad-CAM heatmaps to the web app to show which regions drive each prediction.
Validate on an external chest X-ray dataset.
Compare against plain CNN baselines (for example ResNet50 alone) to measure the contribution of the Transformer encoder.
