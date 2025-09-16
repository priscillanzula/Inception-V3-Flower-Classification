### Inception V3 Flower Classification

A A simple image recognition project using InceptionV3 to classify flower species. Upload a flower photo, and the model predicts: daisy, dandelion, rose, sunflower, or tulip.

###  Features

- Uses pre-trained InceptionV3 (ImageNet) as a base, with a custom classification head.

- Evaluation metrics includes accuracy, precision, recall, and F1-score, plus visual diagnostics like confusion matrices and misclassified examples.

- Strong performance: ~89% overall accuracy; excellent recall for daisy (~95%) and dandelion (~93%); decent but lower recall for tulip (~81%), rose and sunflower.


<img width="691" height="416" alt="per_class_acc" src="https://github.com/user-attachments/assets/af250e21-a5f8-49e7-821d-4ddcc9bc1592" />

<img width="581" height="560" alt="confusion_mat" src="https://github.com/user-attachments/assets/1d0e5657-1f1b-48ed-a731-8e6c704d9db1" />

<img width="235" height="115" alt="Performance" src="https://github.com/user-attachments/assets/28bb7851-3e81-499e-84d6-f020b06feee5" />

Deployable with Streamlit for an interactive demo:


https://github.com/user-attachments/assets/3654c083-0af7-4f30-ae55-d14f286f40ab



### How to Run

1. Clone the repo:

git clone https://github.com/priscillanzula/Inception-V3-Flower-Classification.git
cd Inception-V3-Flower-Classification


2. Install dependencies:

pip install tensorflow streamlit matplotlib scikit-learn


3. To train/evaluate: open image_recognition.ipynb in Jupyter or VSCode, and run the cells in order.

 To deploy the app:

 streamlit run app.py


4. Then upload a flower image via the Streamlit interface for prediction.

### Interpretations 

- Daisy and Dandelion show near-perfect recall, meaning the model almost always correctly identifies them.

- Tulips are often misclassified (especially confused with roses). Roses and sunflowers exhibit more confusion with other classes.

### Contact:
 Priscilla Nzula
