# Fake News Classifier using BERT and Discriminator

This GitHub repository contains code for a Fake News Classifier using the BERT (Bidirectional Encoder Representations from Transformers) model and a Discriminator. The goal of this project is to classify news articles as either "Fake" or "Real" based on their textual content.

## Overview

The code performs the following steps:

1. Load and preprocess the datasets containing fake and real news articles.
2. Combine the datasets and create a balanced dataset for training and testing.
3. Utilize the BERT model for text representation.
4. Implement a Discriminator model to classify the news articles based on the BERT representations.
5. Train the Discriminator model using a custom loss function and optimizer.
6. Evaluate the performance of the trained model on the test dataset.

## Requirements

To run the code, you need to have the following libraries installed:

- Python 3.6 or higher
- Pandas
- PyTorch
- Transformers (from Hugging Face)

You can install the required libraries using `pip`:

pip install pandas torch transformers

## Usage

1. Clone the repository to your local machine.
2. Ensure that the dataset files (`Fake.csv` and `True.csv`) are present in the specified directory (`/content/drive/MyDrive/Colab Notebooks/News _dataset/`).
3. Run the Python script `fake_news_classifier.py`.

The script will load the datasets, preprocess the data, train the Discriminator model, and print the average loss for each epoch.

## Code Explanation

1. **Data Loading and Preprocessing**:
   - The code loads the datasets containing fake and real news articles using Pandas.
   - It adds a label column (`0` for real news, `1` for fake news) to each dataset.
   - The datasets are combined into a single dataframe.
   - The combined dataset is shuffled and split into training and testing sets.

2. **BERT Model**:
   - The BERT model is used for text representation.
   - The code utilizes the `AutoModel` and `AutoTokenizer` from the Transformers library to load the pre-trained BERT model.

   > ## **Why BERT?**
   >
   > **BERT is chosen because it generates contextual embeddings, unlike traditional word embedding models like Word2Vec or GloVe, which use static embeddings. Contextual understanding is crucial for accurately classifying news articles, where the same words can convey different meanings based on the context.**
   >
   > **Static embeddings fail to capture the nuances and subtleties of natural language, leading to potential misinterpretations and inaccuracies in fake news detection, where context is critical. BERT's contextual embeddings enable the Discriminator model to make more informed predictions about the authenticity of news content by effectively encoding contextual cues.**

3. **Discriminator Model**:
   - The Discriminator model is a neural network architecture that takes the BERT representations as input and classifies the news articles as fake or real.
   - The Discriminator consists of multiple convolutional layers followed by a linear layer and a softmax activation function.

4. **Training**:
   - The training process involves iterating over the training dataset in batches.
   - For each batch, the BERT model generates text representations, which are then passed to the Discriminator model.
   - The Discriminator model produces logits and probabilities for the classification task.
   - A custom loss function is calculated based on the log probabilities and the true labels.
   - The Discriminator model is optimized using the Adam optimizer and the calculated loss.
   - The average loss for each epoch is printed.

## Advanced Techniques

This project incorporates several advanced techniques:

1. **Transfer Learning**: The BERT model is a pre-trained transformer model that has been trained on a large corpus of text data. By leveraging transfer learning, the code can take advantage of the knowledge learned by BERT during its pre-training and fine-tune it for the specific task of fake news classification.

2. **Discriminator Model**: The Discriminator model is a custom neural network architecture designed specifically for the task of fake news classification. It takes the BERT representations as input and learns to distinguish between fake and real news articles.

3. **Custom Loss Function**: The code implements a custom loss function that calculates the per-example loss based on the log probabilities and the true labels. This loss function is used to optimize the Discriminator model during training.

4. **Optimization Techniques**: The code utilizes the Adam optimizer, which is an advanced optimization algorithm that combines the advantages of momentum and RMSProp. Additionally, techniques like dropout regularization and LeakyReLU activation functions are employed to improve the performance and prevent overfitting.

## Conclusion

This GitHub repository provides a comprehensive solution for fake news classification using the BERT model and a custom Discriminator architecture. The code incorporates advanced techniques such as transfer learning, custom model architectures, and optimization strategies to achieve accurate and robust classification results. By following the instructions in this README, you can easily run the code and explore the capabilities of this fake news classifier.
