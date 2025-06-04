<b><i><h1> Skin Cancer Detection </h1></i></b>
<img src ="https://miro.medium.com/v2/resize:fit:1400/1*XT9gM3y3rmm4kIOq3l8efw.jpeg" width="600" height="300">
<p>In this project, lesions were being checked for skin cancer and various machine learning models are used for this.</p>
To start with the project,
the dataset used is <i><b><a href= "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T">HAM10000 dataset</a></b></i>
which is a large source of multi source dermatoscopic images of common pigmented skin lesions.
<p>The algorithms used in this code are <ol type="I"><i><li>Yolov5</li><li>Convolutional Neural Network (CNN)</li><li>Elastic Net</li><li>Region-based Convolutional Neural Network (R-CNN)</li><li>Single Shot MultiBox Detector (SSD)</li><li>Deep Neural Network (DNN)</li><li>Bayesian Neural Network (BNN)</li><li>Yolov5-CNN-Hybrid</li><li>RNN-SSD-Hybrid</li><li>DNN-BNN-Hybrid</li><li>Meta Model (combination of the three hybrids)</li></i></ol></p>

In the Main.ipynb, for each algorithm, the code are provided with prediction using a single image and graphical comparisons.

<p>A seperate streamlit code is also provided with the flask implementation for a WebUI and the requirements are also provided, which can be installed by the command "pip install -r requirements.txt".</p>

<p>For accessing all the trained models, a link is provided which redirects to <a href= "https://huggingface.co/AdityaHK/SkinDetect/tree/main"><b><i>Hugging Face</b></i></a> where the trained models can be downloaded/cloned/accessed.</p>
The file directory mapping is given as
  
Main Folder/

│

├── 📓 Main.ipynb               ← Jupyter Notebook for development or testing

├── 🖥️ streamlit_app.py         ← Streamlit frontend application

├── 📄 requirements.txt         ← Required Python packages

│

└── 📁 Dataset                  ← Dataset directory
        
   ├── 📁 Images               ← Folder containing all image files
    
   └── 📄 metadata.csv         ← Metadata describing the images
