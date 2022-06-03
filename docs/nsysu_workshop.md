# 國立中山大學NLP Workshop大綱
- 講師: [Haowen Jiang](https://howard-haowen.rohan.tw/)
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
  
## 第3️⃣週
- 文本向量化1
  - Frequency-based representation 
    - [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- [文本聚類](https://en.wikipedia.org/wiki/Document_clustering)
  - [k-means clustering](https://en.wikipedia.org/wiki/K-means_clustering) 
- Notebook
  - [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/howard-haowen/NLP-demos/blob/main/NSYSU/W03-document-vectorization-and-clustering.ipynb)

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

## 第5️⃣週
- 文本向量化3
  - Dynamic embeddings 
    - [USE](https://tfhub.dev/google/universal-sentence-encoder/4) (Universal Sentence Encoder)
    - [BERT](https://en.wikipedia.org/wiki/BERT_(language_model)) (Bidirectional Encoder Representations from Transformers)
- [文本相似性](https://en.wikipedia.org/wiki/Semantic_similarity#In_natural_language_processing)
- Notebook
  - [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/howard-haowen/NLP-demos/blob/main/NSYSU/W05-transformer-and-document-similarity.ipynb)
  
## 第6️⃣週
- [文本分類](https://en.wikipedia.org/wiki/Document_classification): 傳統機器學習
- 分類算法
  - [Naive Bayes classifiers](https://www.geeksforgeeks.org/naive-bayes-classifiers/?ref=leftbar-rightbar)
  - [Support Vector Machines](https://www.geeksforgeeks.org/support-vector-machine-algorithm/?ref=gcse)
  - [Logistic Regression](https://www.geeksforgeeks.org/understanding-logistic-regression/?ref=gcse)
- Notebook
  - [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/howard-haowen/NLP-demos/blob/main/NSYSU/W06-text-classification-with-scikit-learn.ipynb)

## 第7️⃣週
- [文本分類](https://en.wikipedia.org/wiki/Document_classification): 神經網絡
- 評估指標
  - Accuracy
  - Recall
  - Precision
  - F1
  - [ROC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
- Notebook [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/howard-haowen/NLP-demos/blob/main/NSYSU/W07-text-classification-with-spacy.ipynb)

## 第8️⃣週
- [命名實體](https://en.wikipedia.org/wiki/Named-entity_recognition)
