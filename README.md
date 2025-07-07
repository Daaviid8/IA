# Ruta de Aprendizaje de IA en Castellano

## 1. Machine Learning (ML)
### 1.1 Supervised Learning
- **Algoritmos de Clasificación**
  - Regresión Logística
  - Árboles de Decisión
  - Random Forest
  - Support Vector Machines (SVM)
  - Naive Bayes
  - K-Nearest Neighbors (KNN)
- **Algoritmos de Regresión**
  - Regresión Linear
  - Regresión Polinomial
  - Ridge/Lasso Regression
  - Elastic Net
- **Métricas de Evaluación**
  - Accuracy, Precision, Recall, F1-Score
  - ROC-AUC, Confusion Matrix
  - MSE, RMSE, MAE, R²

### 1.2 Unsupervised Learning
- **Clustering**
  - K-Means
  - Hierarchical Clustering
  - DBSCAN
  - Gaussian Mixture Models
- **Reducción de Dimensionalidad**
  - PCA (Principal Component Analysis)
  - t-SNE
  - UMAP
  - LDA (Linear Discriminant Analysis)
- **Reglas de Asociación**
  - Apriori Algorithm
  - FP-Growth

### 1.3 Reinforcement Learning
- **Conceptos Fundamentales**
  - Agente, Entorno, Estados, Acciones, Recompensas
  - Política, Función de Valor
  - Exploration vs Exploitation
- **Algoritmos Básicos**
  - Q-Learning
  - SARSA
  - Monte Carlo Methods
- **Algoritmos Avanzados**
  - Deep Q-Networks (DQN)
  - Policy Gradient Methods
  - Actor-Critic
  - PPO (Proximal Policy Optimization)

## 2. Neural Networks
### 2.1 Fundamentos de Redes Neuronales
- **Conceptos Básicos**
  - Perceptrón
  - Función de Activación
  - Backpropagation
  - Gradient Descent
- **Arquitecturas Básicas**
  - Multi-Layer Perceptron (MLP)
  - Regularización (Dropout, L1/L2)

### 2.2 Deep Learning (DL)
- **Redes Neuronales Convolucionales (CNN)**
  - Convolución, Pooling, Padding
  - Arquitecturas: LeNet, AlexNet, VGG, ResNet
  - Transfer Learning
- **Redes Neuronales Recurrentes (RNN)**
  - Vanilla RNN, LSTM, GRU
  - Bidirectional RNN
  - Sequence-to-Sequence Models
- **Transformers**
  - Attention Mechanism
  - Self-Attention
  - Multi-Head Attention
  - Encoder-Decoder Architecture
- **Optimización y Técnicas**
  - Adam, RMSprop, SGD
  - Batch Normalization
  - Learning Rate Scheduling

## 3. NLP (Procesamiento de Lenguaje Natural)
### 3.1 Fundamentos de NLP
- **Preprocesamiento de Texto**
  - Tokenización
  - Stemming y Lemmatización
  - Stop Words
  - N-gramas
- **Representación de Texto**
  - Bag of Words
  - TF-IDF
  - Word Embeddings (Word2Vec, GloVe)

### 3.2 Técnicas Avanzadas
- **Modelos de Lenguaje**
  - N-gram Language Models
  - Neural Language Models
- **Análisis Sintáctico y Semántico**
  - POS Tagging
  - Named Entity Recognition (NER)
  - Sentiment Analysis
  - Dependency Parsing

### 3.3 Large Language Models (LLMs)
- **Arquitecturas Transformer**
  - BERT, GPT, T5
  - RoBERTa, DistilBERT
  - GPT-3/4, Claude, LLaMA
- **Fine-tuning y Adaptation**
  - Transfer Learning en NLP
  - Few-shot Learning
  - Prompt Engineering
  - RLHF (Reinforcement Learning from Human Feedback)

## 4. Computer Vision / Machine Vision
### 4.1 Fundamentos de Visión por Computadora
- **Procesamiento de Imágenes**
  - Filtros, Convolución
  - Detección de Bordes
  - Transformaciones Geométricas
- **Extracción de Características**
  - SIFT, SURF, ORB
  - Histogramas de Color
  - Texturas

### 4.2 Técnicas Avanzadas
- **Detección de Objetos**
  - YOLO, R-CNN, Fast R-CNN
  - SSD, RetinaNet
  - Anchor-based vs Anchor-free
- **Segmentación**
  - Semantic Segmentation
  - Instance Segmentation
  - Panoptic Segmentation
  - U-Net, Mask R-CNN
- **Visión 3D**
  - Stereo Vision
  - Structure from Motion
  - 3D Object Detection
  - Point Cloud Processing

## 5. Generative AI
### 5.1 Modelos Generativos Clásicos
- **Generative Adversarial Networks (GANs)**
  - GAN Básico
  - DCGAN, StyleGAN, CycleGAN
  - Conditional GANs
- **Variational Autoencoders (VAE)**
  - Encoder-Decoder Architecture
  - Latent Space Representation
  - β-VAE, VQ-VAE

### 5.2 Modelos de Difusión
- **Diffusion Models**
  - DDPM (Denoising Diffusion Probabilistic Models)
  - DDIM (Denoising Diffusion Implicit Models)
  - Stable Diffusion
- **Aplicaciones**
  - Generación de Imágenes
  - Generación de Video
  - Síntesis de Audio

### 5.3 LLMs Generativos
- **Generación de Texto**
  - GPT Family
  - Text-to-Text Transfer
  - Code Generation
- **Multimodal Generation**
  - Text-to-Image
  - Image-to-Text
  - Video Generation

  
## Herramientas y Frameworks Recomendados

### Programación
- **Python**: NumPy, Pandas, Scikit-learn, Matplotlib
- **Machine Learning**: TensorFlow, PyTorch, Keras
- **NLP**: spaCy, NLTK, Transformers (Hugging Face)
- **Computer Vision**: OpenCV, PIL, Albumentations

### Plataformas y Entornos
- **Cloud**: Google Colab, AWS SageMaker, Azure ML
- **Notebooks**: Jupyter, Google Colab
- **MLOps**: MLflow, DVC, Weights & Biases

### Datasets y Recursos
- **Repositorios**: Kaggle, UCI ML Repository, Papers with Code
- **Datasets**: ImageNet, COCO, MNIST, IMDB Reviews
- **Benchmarks**: GLUE, SuperGLUE, BLEU, ROUGE
