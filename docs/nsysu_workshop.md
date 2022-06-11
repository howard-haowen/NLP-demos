# 國立中山大學NLP Workshop大綱
- 講師: 江豪文[Haowen Jiang](https://howard-haowen.rohan.tw/)
- 時間: 2022-04-15 ~ 2022-06-10
- 大綱內容的[簡報模式](https://hackmd.io/@howard-haowen/nsysu-workshop#/)

## 參考資料
- [spaCy notebooks](https://github.com/explosion/spacy-notebooks)
- [NLP Town notebooks](https://github.com/nlptown/nlp-notebooks)

## 相關書籍
- ![](https://i.gr-assets.com/images/S/compressed.photo.goodreads.com/books/1630086235l/58870327._SX318_.jpg)
- ![](https://i.gr-assets.com/images/S/compressed.photo.goodreads.com/books/1591328063l/53832790._SX318_.jpg)

## 工具
- [nltk](https://www.nltk.org/)
- [spaCy](https://spacy.io/)
  - Playground [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/howard-haowen/rise-env/main?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Fhoward-haowen%252FNLP-demos%26urlpath%3Dtree%252FNLP-demos%252Fspacy_playground.ipynb%26branch%3Dmain)
- [stanza](https://stanfordnlp.github.io/stanza/)
- [gensim](https://radimrehurek.com/gensim/)
- [sklearn](https://scikit-learn.org/stable/)

## 資料集
- [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/howard-haowen/NLP-demos/blob/main/nlp_datasets.ipynb)

## 第1️⃣週
- NLP相關應用
- 熟悉Colab環境
- 調用預訓練模型
  - [Tokenization](https://en.wikipedia.org/wiki/Lexical_analysis#Tokenization)
  - [Parts of speech](https://en.wikipedia.org/wiki/Part_of_speech)
  - [Named entity recognition](https://en.wikipedia.org/wiki/Named-entity_recognition)
  - [Dependency parsing](https://en.wikipedia.org/wiki/Syntactic_parsing_(computational_linguistics)#Dependency_parsing)
- Notebook
  - [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/howard-haowen/NLP-demos/blob/main/NSYSU/W01-use-pretrained-models.ipynb)

> Abstract: In this talk, we showcase some common applications of natural language processing technologies in business. Then we introduce Colab, the coding environment adopted throughout this workshop. Our NLP journey begins with learning how to get access to pretrained NLP models through spaCy for various functionalities, including tokenization, parts of speech tagging, named entity recognition, and dependency parsing.

## 第2️⃣週
- 取得資料集
- 資料預處理
- 訓練主題模型
  - [Bag of Words(BOW)](https://en.wikipedia.org/wiki/Bag-of-words_model)
  - [N-gram](https://en.wikipedia.org/wiki/N-gram)
  - [LDA](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)
  - [HDP](https://en.wikipedia.org/wiki/Hierarchical_Dirichlet_process)
- Notebook
  - [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/howard-haowen/NLP-demos/blob/main/NSYSU/W02-topic-modelling.ipynb)

> Abstract: A topic model in NLP consists of two probability distributions. One has to do with topics over documents and the other with words over topics. In this talk, we go over the pipeline for creating a topic model with Gensim, covering such topics as text preprocessing, bag of words, N-gram, Latent Dirichlet allocation, and Hierarchical Dirichlet process.
    
## 第3️⃣週
- 文本向量化1
  - Frequency-based representation 
    - [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- [文本聚類](https://en.wikipedia.org/wiki/Document_clustering)
  - [k-means clustering](https://en.wikipedia.org/wiki/K-means_clustering) 
- Notebook
  - [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/howard-haowen/NLP-demos/blob/main/NSYSU/W03-document-vectorization-and-clustering.ipynb)

> Abstract: Text clustering is an unsupervised way of assigning texts to clusters. In this talk, we go over the pipeline for building a text clustering model using the K-means algorithm, where K represents a predefined number of clusters. One prerequisite of implementing K-means is vectorization of texts. We will learn how to use TF-IDF as a baseline approach to text vectorization. 

## 第4️⃣週
- 文本向量化2
  - Static word embeddings 
    - [Word2vec](https://en.wikipedia.org/wiki/Word2vec) by Google
    - [fastText](https://en.wikipedia.org/wiki/FastText) by Facebook
    - [GloVe](https://en.wikipedia.org/wiki/GloVe) (Global Vectors) by the Stanford NLP team
- [文本聚類](https://en.wikipedia.org/wiki/Document_clustering)
  - [k-means clustering](https://en.wikipedia.org/wiki/K-means_clustering) 
- Notebook
  - 沿用第3️⃣週的Notebook

> Abstract: Last week, we implemented the K-means algorithm for text clustering by vectorizing texts with TF-IDF. This week, we continue to experiment with text clustering, but with more complicated approaches to text vectorization, namely word embeddings. Common architectures for word embeddings include Word2vec by Google, fastText by Facebook, and Glove by the Stanford NLP team. We will train a fastText embedding model on the fly with Gensim and employ the same K-means algorithm as from last week.

## 第5️⃣週
- 文本向量化3
  - Dynamic embeddings 
    - [USE](https://tfhub.dev/google/universal-sentence-encoder/4) (Universal Sentence Encoder)
    - [BERT](https://en.wikipedia.org/wiki/BERT_(language_model)) (Bidirectional Encoder Representations from Transformers)
- [文本相似性](https://en.wikipedia.org/wiki/Semantic_similarity#In_natural_language_processing)
- Notebook
  - [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/howard-haowen/NLP-demos/blob/main/NSYSU/W05-transformer-and-document-similarity.ipynb)

> Abstract: While word embedding models like Word2vec, fastText, and Glove are powerful, they are essentially large lookup tables that consistently map a linguistic token to a fixed-length dense vector, thus failing to capture more nuanced information from context. This week, we leverage pretrained models for dynamic embeddings to build a vector-based search engine of texts, including Google's Universal Sentence Encoder (USE) and BERT ((Bidirectional Encoder Representations from Transformers), which dynamically calculate vectors from context given a stretch of text. To facilitate the vector-based search, we tap into Facebook's FAISS library to create an embedding index, which makes it much faster to search for similar vectors.

## 第6️⃣週
- [文本分類](https://en.wikipedia.org/wiki/Document_classification): 傳統機器學習
- 分類算法
  - [Naive Bayes classifiers](https://www.geeksforgeeks.org/naive-bayes-classifiers/?ref=leftbar-rightbar)
  - [Support Vector Machines](https://www.geeksforgeeks.org/support-vector-machine-algorithm/?ref=gcse)
  - [Logistic Regression](https://www.geeksforgeeks.org/understanding-logistic-regression/?ref=gcse)
- Notebook
  - [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/howard-haowen/NLP-demos/blob/main/NSYSU/W06-text-classification-with-scikit-learn.ipynb)

> Abstract: Text classification is a prominent example of supervised learning in NLP, whereby texts are automatically assigned a category. There are numerous use cases for text classification, including email spam detector, hate speech detector, customer sentiment analysis, customer support system, news classification, and even chatbot intent classification. This week, we go through the steps for training text classification models using traditional machine learning methods, such as TF-IDF vectorizer, Naive Bayes classifiers, Support Vector Machines, and Logistic Regression. We also look into how to evaluate the trained models and explain why they work in the first place.

## 第7️⃣週
- [文本分類](https://en.wikipedia.org/wiki/Document_classification): 神經網絡
- 評估指標
  - Accuracy
  - Recall
  - Precision
  - F1
  - [ROC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
- Notebook [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/howard-haowen/NLP-demos/blob/main/NSYSU/W07-text-classification-with-spacy.ipynb)

> Abstract: Last week, we trained text classification models in traditional machine learning methods using Scikit-learn. This week, we move one step forward to carry out the same task but leverage the power of neural networks using spaCy. It is shown that training and evaluation can be highly streamlined, replicable, and efficient when they are done in spaCy's command lines. While evaluating classification models, we also introduce concepts such as Receiver operating characteristic (ROC) and Area Under the ROC Curve (AUC).

## 第8️⃣週
- [命名實體](https://en.wikipedia.org/wiki/Named-entity_recognition)
- Notebook [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/howard-haowen/NLP-demos/blob/main/NSYSU/W08-extracting-named-entities-with-spacy.ipynb)

> Abstract: Named entity recognition (NER) is a highly valuable AI capability and widely used in industries like ecommerce, social media, and FinTech. This week, we show how to train and evaluate NER models using spaCy's command lines, which is a process very similar to what we did last week while training text classification models. We then compare the performance of the newly trained model with that of a pretrained spaCy model.

[Back to top](https://howard-haowen.rohan.tw/NLP-demos/nsysu_workshop)
