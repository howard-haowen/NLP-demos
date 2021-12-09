# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] id="view-in-github" colab_type="text"
# <a href="https://colab.research.google.com/github/howard-haowen/Chinese-NLP/blob/main/NQU_talk.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% id="QEuBPlA2bXZn"
from traitlets.config.manager import BaseJSONConfigManager
from pathlib import Path

# %% id="MJFMGDwcba-I"
path = Path.home() / ".jupyter" / "nbconfig"
cm = BaseJSONConfigManager(config_dir=str(path))
cm.update(
    "rise",
    {
        "autolaunch": True,
        "enable_chalkboard": True,
        "scroll": True,
        "slideNumber": True,
        "controls": True,
        "progress": True,
        "history": True,
        "center": True,
        "width": "90%",
        "height": "90%",
        "theme": "beige",
        "transition": "concave",
        "start_slideshow_at": "selected"
     }
)

# %% [markdown] id="qtGZFvKvJ87Y"
# 機器學習基本流程
#
# ![](https://2s7gjr373w3x22jf92z99mgm5w-wpengine.netdna-ssl.com/wp-content/uploads/2018/09/WD_3.png)

# %% [markdown] id="SCSIja8pw63y"
# # 下載資料 

# %% id="Kn5vW2nqfuim"
import pandas as pd

# %% colab={"base_uri": "https://localhost:8080/"} id="itGmQQ13qIHz" outputId="17dd73ec-fa61-4bb8-fb5d-e223fd5fa5be"
data_path = "online_shopping_5_cats_tra.csv"
df = pd.read_csv(data_path)
df.sample(5, random_state=100)

# %% colab={"base_uri": "https://localhost:8080/"} id="23Zmj5scv1bn" outputId="28ab322a-08ca-47da-a90e-3e20694f72ca"
df['cat'].value_counts()

# %% colab={"base_uri": "https://localhost:8080/"} id="idwOSCFmv2i8" outputId="8cd31bc8-a453-451b-f025-e338cc3fe57e"
df['label'].value_counts()

# %% colab={"base_uri": "https://localhost:8080/"} id="RLb8qaGi7-AE" outputId="96fef858-ca43-49c8-c054-4076e666dbcb"
# %load_ext google.colab.data_table
from google.colab import data_table
data_table.DataTable(df, include_index=True, num_rows_per_page=10)

# %% [markdown] id="U8ud-0_KxBWz"
# # 基本語言分析

# %% [markdown] id="Y9YaJKatJ12e"
# ## 斷詞

# %% id="6KLrYo3k8bH-" colab={"base_uri": "https://localhost:8080/"} outputId="fad7da31-ea6f-4168-c721-df6eb3355ed2"
# !pip install -q -U pip setuptools wheel 
# !pip install -q -U spacy
# !python -m spacy download zh_core_web_md

# %% colab={"base_uri": "https://localhost:8080/"} id="ztHzKCBAfyru" outputId="42333bb8-0a6a-4b38-bfbb-a2352c719a16"
import spacy

nlp = spacy.load("zh_core_web_md")
nlp.pipe_names

# %% colab={"base_uri": "https://localhost:8080/", "height": 35} id="naIUelyaQSva" outputId="a42b5dcb-50eb-4236-f954-450c60a6cf00"
text = df.at[10528, 'review']
text

# %% id="G-Z040z2J9-R"
text = df.at[10528, 'review']
doc = nlp(text)
tokens = [tok.text for tok in doc]
" | ".join(tokens)

# %% colab={"base_uri": "https://localhost:8080/"} id="ZaY5RPFpT8Fq" outputId="0a6fd019-25ff-46fa-ec40-2a656794db1e"
# Clone the entire repo.
# !git clone -l -s https://github.com/L706077/jieba-zh_TW.git jieba_tw
# %cd jieba_tw
# !ls

# %% id="ACh_KTpsUmhC"
import jieba

# %% colab={"base_uri": "https://localhost:8080/", "height": 102} id="UVAhkqRyVGSr" outputId="082f9f90-35e5-47f8-f618-addb96b6edb8"
text = df.at[10528, 'review']
tokens = jieba.cut(text) 
" | ".join(tokens)

# %% [markdown] id="veULoMBuyvmw"
# ## 依存句法

# %% id="BREfnaQcaQ3W"
from spacy import displacy

# %% colab={"base_uri": "https://localhost:8080/"} id="U32pNPApXs4C" outputId="d43972f4-7f67-470c-d0ea-2e2710440f12"
text = """國立金門大學第1屆樂齡大學今天開學，30名長者展開36週的學習。"""
doc = nlp(text)
displacy.render(doc, style='dep',jupyter=True, options={'distance':130})

# %% [markdown] id="-W4sPfcZYMCq"
# ## 命名實體

# %% colab={"base_uri": "https://localhost:8080/"} id="KozpzTVNYLnb" outputId="7bd93ee7-22e8-469e-a6a9-74a390369752"
text = """
顏達仁曾是金門第1任副縣長，
今年進入金大進修部企管系1年級就讀，
不僅是本屆最高齡，
也是金大創校以來，
年紀最長的大學生。
"""
doc = nlp(text)
displacy.render(doc, style='ent',jupyter=True)

# %% id="DOXM8H_JJqgT"
# !pip install -q dframcy

# %% id="t8Pnsbb1Jk9w"
from dframcy import DframCy

dframcy = DframCy(nlp)

# %% colab={"base_uri": "https://localhost:8080/"} id="qcTx_de5JZhZ" outputId="b22596af-8d04-4c39-b88a-4e0627de84d4"
sample = df.at[9999, 'review']
doc = dframcy.nlp(sample)
annotation_dataframe = dframcy.to_dataframe(doc)
annotation_dataframe

# %% [markdown] id="qzfcM7s49n0a"
# ## 詞頻統計

# %% colab={"base_uri": "https://localhost:8080/"} id="_jXpbG52i1it" outputId="ad21c18f-5d3f-4205-f2cb-d130ab2bdf8f"
# !pip install -q scattertext

# %% id="k1zeldi_jDFr"
import scattertext as st

# %% colab={"base_uri": "https://localhost:8080/"} id="wMO1L-2UjLUg" outputId="34a6bb58-75a8-4a3f-98e0-292cfa6252ac"
df['cat'].unique()

# %% colab={"base_uri": "https://localhost:8080/"} id="qwtWcHc7jXXF" outputId="0309ca2d-0cbb-4ca9-fde7-64e0ea1b2641"
filt = (df['cat'] == '水果') | (df['cat'] == '洗髮水')
sample_data = df[filt].sample(500, random_state=100)
sample_data

# %% id="CbTD1Q6vi-HH"
corpus = st.CorpusFromPandas(sample_data,
                             category_col='cat',
                             text_col='review',
                             nlp=nlp).build()

# %% colab={"base_uri": "https://localhost:8080/"} id="mpGZ1MGj6mhq" outputId="d3141324-f846-4338-f602-67fc3cfe6f21"
term_freq_df = corpus.get_term_freq_df()
term_freq_df

# %% id="M8cMi5Dy7Kq_"
html = st.produce_scattertext_explorer(corpus, 
                                       category='水果', 
                                       category_name='水果', 
                                       not_category_name='洗髮水',
                                       width_in_pixels=1000, 
                                       metadata=sample_data['label'])

#open("scattertext_01.html",'wb').write(html.encode('utf-8'))

# %% colab={"base_uri": "https://localhost:8080/"} id="UYbmjHQN72YD" outputId="74c06cd0-974a-4a83-a68e-5a720bf5c11a"
from IPython.display import display, HTML

display(HTML(html))

# %% colab={"base_uri": "https://localhost:8080/"} id="ChGxFW-I-HEa" outputId="a91c40b6-5ae2-4152-e760-9db19627ff42"
filt = (df['cat'] == '水果') | (df['cat'] == '平板')
sample_data = df[filt].sample(500, random_state=100)
sample_data

# %% colab={"base_uri": "https://localhost:8080/"} id="SPZ5F8n6-Sqs" outputId="c18492b9-0b49-4538-d22a-30e0b56c3270"
corpus = st.CorpusFromPandas(sample_data,
                             category_col='cat',
                             text_col='review',
                             nlp=nlp).build()
html = st.produce_scattertext_explorer(corpus, 
                                       category='水果', 
                                       category_name='水果', 
                                       not_category_name='平板',
                                       width_in_pixels=1000, 
                                       metadata=sample_data['label'])
display(HTML(html))

# %% [markdown] id="m8cb4vQwxp3W"
# # 文本預處理

# %% colab={"base_uri": "https://localhost:8080/"} id="A8NafXzPC8tX" outputId="59698e96-f178-4684-b3bb-10f2ffb21d55"
text = """
顏達仁曾是金門第1任副縣長，
今年進入金大進修部企管系1年級就讀，
不僅是本屆最高齡，
也是金大創校以來，
年紀最長的大學生。
"""
text

# %% colab={"base_uri": "https://localhost:8080/"} id="jO6sGjxoEMcV" outputId="1319edb0-dd14-4ba7-aaed-84a46b4cc203"
text = df.at[31902, 'review']
text

# %% id="Ql3deDTDMDE1"
from spacy.tokens import Doc

class WhitespaceTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.strip().split()
        return Doc(self.vocab, words=words)

nlp = spacy.load("zh_core_web_md", disable=["tagger", "parser", "ner"])
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)


# %% id="NltTjnFAMekf"
def whitespace_cut(raw_text: str) -> str:
    tokens = jieba.cut(raw_text)
    results = " ".join(tokens)
    return results


# %% id="QhLoytocMUc_"
def preprocess_text(raw_text: str) -> str:
    raw_text_with_space = whitespace_cut(raw_text)
    doc = nlp.make_doc(raw_text_with_space)
    res = [tok for tok in doc if not tok.is_punct]
    res = [tok for tok in res if not tok.is_stop]
    res = [tok for tok in res if not tok.like_email]
    res = [tok for tok in res if not tok.like_url]
    res = [tok for tok in res if not tok.like_num]
    res = [tok for tok in res if not tok.is_ascii]
    res = [tok.text for tok in res if not tok.is_space]
    res = " ".join(res)
    return res


# %% colab={"base_uri": "https://localhost:8080/"} id="HMYJ1RC5GbLk" outputId="e29998c3-e332-4c48-f97e-004dafa3beae"
tokenized_df = df.sample(1000, random_state=100)
tokenized_df['tokenized_text'] = tokenized_df['review'].apply(preprocess_text)
tokenized_df

# %% [markdown] id="w7MHZyDCJyWM"
# # 文本表徵

# %% [markdown] id="_j3urOZK-eDh"
# 電腦的世界只有數字
#
# ![](data:image/jpeg;base64,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)

# %% [markdown] id="OkXUDKlaJ7Z_"
# ## TF-IDF

# %% [markdown] id="l2u951-CQKxA"
# ![](https://miro.medium.com/max/1000/1*vWWmJlDykVRkjg9c38VbxQ.png)

# %% colab={"base_uri": "https://localhost:8080/"} id="4RzA97fLWdOn" outputId="96604c5a-0887-42e7-c5da-f3a401d640d1"
# !pip install -q -U texthero

# %% colab={"base_uri": "https://localhost:8080/"} id="L54vkQgVKDrX" outputId="1dbcd089-d551-42e8-9143-68d35f2656bb"
import texthero as hero

# %% colab={"base_uri": "https://localhost:8080/"} id="MiWOGGwoKIAV" outputId="f6ddb50a-9b6d-426c-fa59-2ba8a3ec2505"
tokenized_df['tfidf_text'] = hero.tfidf(tokenized_df['tokenized_text'])
tokenized_df[['cat', 'tokenized_text', 'tfidf_text']]

# %% colab={"base_uri": "https://localhost:8080/"} id="BMxKRQ8HslkD" outputId="d315a257-943b-457a-8599-043a101090a0"
len(tokenized_df.at[40316, 'tfidf_text'])

# %% [markdown] id="pEMk2GHyLIAO"
# ## PCA

# %% [markdown] id="FS_zJujkRv8z"
# ![](https://3.bp.blogspot.com/-aGt6JrHLt_E/WjXd4Ge7B-I/AAAAAAAAAgY/-gF8sXb4qqAwObOmP9hTiJa0ot5wR82ZACLcBGAs/s1600/Principal-Component-Analysis-And-Dimensionality-Reduction.jpg)

# %% colab={"base_uri": "https://localhost:8080/"} id="5_5we3k3LIeH" outputId="3b0f543a-587d-4daf-ced0-6a933e00356c"
tokenized_df['pca_tfidf_text'] = hero.pca(tokenized_df['tfidf_text'])
tokenized_df[['cat', 'tokenized_text', 'pca_tfidf_text']]

# %% colab={"base_uri": "https://localhost:8080/"} id="gIvsvZilMY-R" outputId="dc822235-7313-4b9a-f72d-dda82ae29fd3"
hero.scatterplot(tokenized_df, col='pca_tfidf_text', color='cat', title="商品評論PCA")

# %% colab={"base_uri": "https://localhost:8080/"} id="B8ZU0ELGM132" outputId="2824b302-347e-4e44-e3cb-1a12107a762e"
NUM_TOP_WORDS = 10
tokenized_df.groupby('cat')['tokenized_text'].apply(lambda x: hero.top_words(x)[:NUM_TOP_WORDS])

# %% [markdown] id="9VrjxDV1Ounm"
# 意義如何量化?
#
# ![](https://slideplayer.com/slide/12147948/71/images/10/John+Rupert+Firth+You+shall+know+a+word+by+the+company+it+keeps.jpg)
#

# %% [markdown] id="mYI4Y01bOksK"
# ## fasttext詞向量

# %% [markdown] id="T452ZOBJTyy-"
# ![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTGk4DaFJFVjMkBp3Rhj9Mbheeo_RJ_BBLdZUv45dAqurXHHWpa1I_bnfE5lYDL86WmREU&usqp=CAU)

# %% [markdown] id="C6TtVLa0T6Dr"
# ![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSz3B9QMK0cWCuJTLWjFz-qoQxD2acPXSGcdw&usqp=CAU)

# %% colab={"base_uri": "https://localhost:8080/"} id="snyTJjP-OjyZ" outputId="8a0c0bac-9acb-4064-9dcf-bb93205519c0"
# !pip install -q fasttext

# %% id="1ZNISc8cW_Ob"
import fasttext

# %% colab={"base_uri": "https://localhost:8080/"} id="5o142nccV8Z2" outputId="e4de1143-ccb0-4388-e582-f2cd5f03600e"
texts = tokenized_df['tokenized_text'].to_list()
texts[:5]

# %% id="q1rLMwRQWSsG"
with open('corpus.txt', 'w') as f:
    for item in texts:
        f.write(f"{item}\n")

# %% id="hG8jsqzcW3vX"
model = fasttext.train_unsupervised('corpus.txt')

# %% colab={"base_uri": "https://localhost:8080/"} id="GS8x3jjbX5LO" outputId="67885516-b393-4f7b-ada6-4690aab8dd50"
len(model.words)

# %% colab={"base_uri": "https://localhost:8080/"} id="FPtM1vX8XGg-" outputId="474875e7-9c37-4c17-a232-63d1907e7870"
model.words[:20]

# %% colab={"base_uri": "https://localhost:8080/"} id="sailjyr4XN-x" outputId="d6db8656-7e25-41b6-cb6f-bd1478d03ab6"
model.get_nearest_neighbors('雙十一')

# %% colab={"base_uri": "https://localhost:8080/"} id="-MC9pic_tJue" outputId="ffba9ce7-c745-4072-9c0b-d1969dd99533"
model.get_nearest_neighbors('總的來說')

# %% [markdown] id="eSVnZT7zbQKM"
# ## word2vec詞向量

# %% colab={"base_uri": "https://localhost:8080/"} id="52OiKCh1bVxI" outputId="8eaf3937-6ebc-4557-9585-23b7eff94831"
# !pip install -q whatlies

# %% id="4Frv8hiGbexz"
from whatlies import EmbeddingSet
from whatlies.language import SpacyLanguage

lang = SpacyLanguage('zh_core_web_md')

# %% colab={"base_uri": "https://localhost:8080/"} id="9-Jadg3tcIwk" outputId="fbbb7f0d-b36e-4aa7-94f1-a7516d0944d9"
words = ['醫生', '護士', '醫院', '工程師', '經理', '公司']
emb = lang[words]
emb.plot_interactive(x_axis='醫院', y_axis='公司')

# %% [markdown] id="9ZzgHzxmL2Ha"
# [向量投影視覺化](https://projector.tensorflow.org/)

# %% [markdown] id="KnPkbsmIOWtx"
# # 文本相似性

# %% id="GXcHHAGOgp-A"
nlp = spacy.load("zh_core_web_md")

# %% colab={"base_uri": "https://localhost:8080/"} id="__vUOwVVf9D2" outputId="c67db54c-9e39-481f-8a92-da267e514e60"
text01 = nlp("我就讀金門大學")
text02 = nlp("我在金大上學")
text03 = nlp("我喜歡語言學")

sim01 = text01.similarity(text02)
sim02 = text02.similarity(text03)

print(sim01)
print(sim02)

# %% [markdown] id="HSL9HYTKuFaS"
# 餘弦相似性
#
# ![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQHFjinj2_lYY-tQ9rTzOiKpwovrfzJ60QXPg&usqp=CAU)

# %% colab={"base_uri": "https://localhost:8080/"} id="AQrx7oKAMPon" outputId="b146b30e-65f4-4cc4-f481-253b1181a2d0"
# !pip install -q faiss-cpu

# %% id="_eVx5XTvMSJZ"
import faiss

# %% id="4jWRvSNtMjnn"
sample_df = df.sample(500,random_state=100)
text_list = sample_df['review'].to_list()
index_list = sample_df.index.values

# %% colab={"base_uri": "https://localhost:8080/"} id="xagAoKlNEzXi" outputId="055f85cf-e5b8-448f-bf7d-2c85714c9a3c"
text_list[-3:]

# %% id="KKFoYtZBM5Tp"
import numpy as np

def create_index_embeddings(doc_vectors, index_series):
    # Step 1: Change data type
    embeddings = np.array([embedding for embedding in doc_vectors]).astype("float32")

    # Step 2: Instantiate the index
    index = faiss.IndexFlatL2(embeddings.shape[1]) 

    # Step 3: Pass the index to IndexIDMap
    index = faiss.IndexIDMap(index)

    # Step 4: Add vectors and their IDs
    index.add_with_ids(embeddings, index_series)

    return index, embeddings


# %% colab={"base_uri": "https://localhost:8080/"} id="z2yYv3ycOJm2" outputId="bf129756-f0e9-446e-e86b-14217e8f37ef"
text_list[-1]

# %% id="dYv6bsmAMY_v"
doc_vectors = []

for doc in nlp.pipe(text_list):
    doc_vectors.append(doc.vector)

index, embeddings = create_index_embeddings(doc_vectors, index_list)

# %% id="xZ6lD1QFQGYG"
D, I = index.search(np.array([embeddings[499]]), k=5)
SNs = I.flatten().tolist()

# %% colab={"base_uri": "https://localhost:8080/"} id="2nBfqs1MQiD9" outputId="eade6d0c-33b6-476e-f82a-eb8514fdec0e"
sample_df.loc[SNs, :]

# %% [markdown] id="-HD7iSDvjK2f"
# # 文本分類

# %% colab={"base_uri": "https://localhost:8080/"} id="Yqch4JdPwfHE" outputId="2d37b0ff-5df1-4340-f319-6670b867a9df"
train_df = df.sample(frac=0.2,random_state=100)
train_df['label'].value_counts()

# %% id="s_OZfgJtwG2D"
from spacy.pipeline.textcat_multilabel import DEFAULT_MULTI_TEXTCAT_MODEL

# %% id="cAafawsOwKP7"
config = {
"threshold": 0.5,
"model": DEFAULT_MULTI_TEXTCAT_MODEL
}
textcat = nlp.add_pipe("textcat_multilabel", config=config)

# %% id="5D8Oy43dx1RU"
from spacy.training import Example

# %% id="yz2XLi_DxgbR"
train_examples = []
for index, row in train_df.iterrows():
    text = row["review"]
    rating = row["label"]
    label = {"正面": True, "負面": False} if rating == 1 else {"負面": True, "正面": False}
    train_examples.append(Example.from_dict(nlp.make_doc(text), {"cats": label}))

# %% id="cb3hOl73yFcX"
textcat.add_label("正面")
textcat.add_label("負面")
textcat.initialize(lambda: train_examples, nlp=nlp)

# %% id="IajtZqhfygrG"
import random

# %% colab={"base_uri": "https://localhost:8080/"} id="yJ9rHQSVyQU2" outputId="79b1006f-0d2d-46fb-e21f-f638453d5c6d"
epochs = 2
with nlp.select_pipes(enable="textcat_multilabel"):
    optimizer = nlp.resume_training()
    for i in range(epochs):
        random.shuffle(train_examples)
        for example in train_examples:
            nlp.update([example], sgd=optimizer)

# %% colab={"base_uri": "https://localhost:8080/"} id="o90PCYO60bKb" outputId="7f08d0f7-0fce-432a-eee1-d407b156e42c"
text = "這種假東西還敢賣這麼貴"
doc = nlp(text)
doc.cats

# %% colab={"base_uri": "https://localhost:8080/"} id="f53LoQbN0tpn" outputId="344544e1-458b-4256-d537-6aae1d2f97ed"
train_indexes = train_df.index.to_list()
test_indexes = [idx for idx in df.index.to_list() if idx not in train_indexes]
test_indexes[-5:]

# %% colab={"base_uri": "https://localhost:8080/", "height": 136} id="YBI5erxW25H5" outputId="a683b8d5-ce31-4a33-a3ca-4aca2be1a19a"
text = df.at[49999, 'review']
sentiment = df.at[49999, 'label']
print(sentiment)
text

# %% colab={"base_uri": "https://localhost:8080/"} id="-D94DeedIQGI" outputId="77f591ba-4369-4780-f97f-60ee2f273aee"
doc = nlp(text)
doc.cats


# %% id="7NMeFWm5IU34"
def show_test():
    idx = random.choice(test_indexes)
    text = df.at[idx, 'review']
    sentiment = df.at[idx, 'label']
    predict = nlp(text).cats
    print(f"編號:{idx}")
    print(f"評論:{text}")
    print(f"正確情感:{sentiment}")
    print(f"模型判斷:{predict}")


# %% colab={"base_uri": "https://localhost:8080/"} id="UhMaxlewJ41G" outputId="972a82fc-1d5e-44ec-9349-b3620f8d4b9a"
show_test()

# %% id="OL6hmOmLDH3A"
# # !pip install RISE jupyter_contrib_nbextensions
# # !jupyter contrib nbextension install --system

# %% id="lVv9AUbEDqfN"
# # !pip install pyngrok --quiet

# %% colab={"base_uri": "https://localhost:8080/"} id="Ie9O2QJXEL9n" outputId="4057e76c-a897-4d56-c478-048a02605ac4"
"""
from pyngrok import ngrok

ngrok.kill()
auth_token = ""
ngrok.set_auth_token(auth_token)
"""

# %% id="-rs0cxc9EWTc"
# get_ipython().system_raw('jupyter notebook NLPDemo.ipynb --ip=0.0.0.0 --port=1400 &')

# %% id="zh-ZGu1kE4k4"
# ngrok.connect(1400)
