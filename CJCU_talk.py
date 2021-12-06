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

# %% [markdown] colab_type="text" id="view-in-github" slideshow={"slide_type": "skip"}
# <a href="https://colab.research.google.com/github/howard-haowen/Chinese-NLP/blob/main/NLP_talk.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% id="QEuBPlA2bXZn" slideshow={"slide_type": "skip"}
from traitlets.config.manager import BaseJSONConfigManager
from pathlib import Path

# %% colab={"base_uri": "https://localhost:8080/"} id="MJFMGDwcba-I" outputId="26117279-9541-4302-8d2a-72ae49f3f0b5" slideshow={"slide_type": "skip"}
path = Path.home() / ".jupyter" / "nbconfig"
cm = BaseJSONConfigManager(config_dir=str(path))
cm.update(
    "rise",
    {
        "autolaunch": False,
        "enable_chalkboard": True,
        "scroll": True,
        "slideNumber": True,
        "controls": True,
        "progress": True,
        "history": True,
        "center": True,
        "width": "100%",
        "height": "100%",
        "theme": "beige",
        "transition": "concave",
        "start_slideshow_at": "selected"
     }
)

# %% [markdown] id="0s3kuqUKPZta" slideshow={"slide_type": "slide"}
# # 自我介紹

# %% [markdown] id="lm5dm2XyaZ9c"
# ## 教育背景
# ![](https://raw.githubusercontent.com/howard-haowen/Chinese-NLP/main/education.png)

# %% [markdown] id="KxN1aUc4kr18"
# ## 現職
#
# - 香港商慧科訊業台灣分公司AI工程師
#
# ![](https://raw.githubusercontent.com/howard-haowen/Chinese-NLP/main/wisers_ai_lab.png)

# %% [markdown] id="hb4xGyyNslMd" slideshow={"slide_type": "-"}
# # 機器學習基本流程
#
# ![](https://2s7gjr373w3x22jf92z99mgm5w-wpengine.netdna-ssl.com/wp-content/uploads/2018/09/WD_3.png)

# %% [markdown] id="dieJ3Nq4eiSb" slideshow={"slide_type": "slide"}
# # 大綱
# - 基本NLP介紹
#  * 商業應用
#  * 開發工具
#  * 基本語言分析
# - 文本表徵
#  * 象徵式表徵(symoblic representations)
#  * 分佈式表徵(distribut**ional** representations)
#  * 分散式表徵(distribut**ed** representations)
#
# -  遷移學習
#  * 領域遷移(domain adaptation)
#  * 語言遷移(cross-lingual learning) 
#  * 任務遷移(multi-task learning)
#

# %% [markdown] id="3a3ip6WWSHsA"
# ## 參考書籍
#
# <figure>
#  <img src="https://i.gr-assets.com/images/S/compressed.photo.goodreads.com/books/1630086235l/58870327._SX318_.jpg" width:50%>
#   <img src="https://i.gr-assets.com/images/S/compressed.photo.goodreads.com/books/1628096299l/55711023._SX318_.jpg" width:50%>
# </figure>

# %% [markdown] id="YgmpViOuK4Z4" slideshow={"slide_type": "slide"}
# # 常見NLP商業應用
#
# ![](https://camo.githubusercontent.com/7eb96ba311f989fbac5257f412a7f454f695015e7ce52d29ac24e10c8f34ef9e/68747470733a2f2f7777772e63796269616e742e636f6d2f77702d636f6e74656e742f75706c6f6164732f323032302f30312f434b432d4e61747572616c2d4c616e67756167652d50726f63657373696e672e706e67)

# %% [markdown] id="a73sgPLZhg7i"
# ## 全球NLP營收預估
#
# ![](https://github.com/howard-haowen/Chinese-NLP/raw/main/nlp_revenues.png)

# %% [markdown] id="aBB0Kc10LFxq"
# ## 模型能力展示：[Ｗisers AI Lab](https://www.wisers.ai/zh-hk/browse/single-article-analysis/)
#
# - 命名實體識別
# - 情感分析
# - 話題分類
# - 關係抽取
# - 評論歸納
#

# %% [markdown] id="U2q0HQipNwyy" slideshow={"slide_type": "slide"}
# # NLP開發工具

# %% [markdown] id="Us8FwcHEwJuX"
# ## 基本套件
# ![](https://lh4.googleusercontent.com/5WioE0xqmDCRydcfrYXaTDUOewQzQXbFmehGVTk5IeLiN3uTeDuCtPrPhedE-3X8ghElL0Z2McxganYCvGWhufZR9xEgKkAQL1pyC1d5E3CnY7rH_c7iA2XKBzpkagnKK5TBf54c)

# %% [markdown] id="_GaxajzGlass"
# ## 進階套件
#
#  * [HuggingFace](https://huggingface.co/models) transformers
#  * PyTorch
#  * TensorFlow
#
# ![](https://venturebeat.com/wp-content/uploads/2019/09/hugging-face.png?w=1200&strip=all)

# %% [markdown] id="U8ud-0_KxBWz" slideshow={"slide_type": "slide"}
# # 基本語言分析

# %% [markdown] id="Y9YaJKatJ12e" slideshow={"slide_type": "subslide"}
# ## 斷詞

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="6KLrYo3k8bH-" outputId="9c3ca37b-c227-4044-bc56-be85134d27ee"
# !pip install -U pip setuptools wheel
# !pip install -U spacy
# !python -m spacy download zh_core_web_md

# %% colab={"base_uri": "https://localhost:8080/"} id="PuXjrsVniTAf" outputId="5f8317d2-d196-47cd-e43d-812d25e747c0"
# !pip show spacy

# %% colab={"base_uri": "https://localhost:8080/"} id="ztHzKCBAfyru" outputId="ce35d535-1d76-4c7e-d557-2fcb03ac3f26"
import spacy

nlp = spacy.load("zh_core_web_md")
nlp.pipe_names

# %% [markdown] id="aLxNcrXGO7SP"
# ### spaCy的模組化設計
# ![](https://spacy.io/pipeline-fde48da9b43661abcdf62ab70a546d71.svg)

# %% [markdown] id="MzOcpL4yPvF6"
# ### spaCy的內建斷詞模型

# %% colab={"base_uri": "https://localhost:8080/", "height": 36} id="G-Z040z2J9-R" outputId="4059e3db-0928-47a2-d9df-fbbb1536105b"
text = "送人的，不知道好不好。聽說紅心火龍果還是比較甜的。"
doc = nlp(text)
tokens = [tok.text for tok in doc]
" | ".join(tokens)

# %% colab={"base_uri": "https://localhost:8080/", "height": 36} id="JU7p1GVvxSFB" outputId="29cc4996-af45-4f4d-9303-303f9d2ec1b9"
text = "送人的，不知道好不好。听说红心火龙果还是比较甜的。"
doc = nlp(text)
tokens = [tok.text for tok in doc]
" | ".join(tokens)

# %% [markdown] id="xvUPwUo8Ptad"
# ### 繁體中文版jieba

# %% colab={"base_uri": "https://localhost:8080/"} id="ZaY5RPFpT8Fq" outputId="c518e2ba-f156-44ef-a441-c7e04160313d"
# !git clone -l -s https://github.com/L706077/jieba-zh_TW.git jieba_tw
# %cd jieba_tw
# !ls

# %% id="ACh_KTpsUmhC"
import jieba

# %% colab={"base_uri": "https://localhost:8080/", "height": 36} id="UVAhkqRyVGSr" outputId="697a6f1e-e50f-4884-f388-2ad1bfb9467b"
text = "送人的，不知道好不好。聽說紅心火龍果還是比較甜的。"
tokens = jieba.cut(text) 
" | ".join(tokens)

# %% colab={"base_uri": "https://localhost:8080/", "height": 36} id="VnQCbHsxKini" outputId="c7f98f1b-194e-4a5f-ad02-b42c159737e0"
text = "送人的，不知道好不好。听说红心火龙果还是比较甜的。"
tokens = jieba.cut(text) 
" | ".join(tokens)

# %% [markdown] id="-W4sPfcZYMCq" slideshow={"slide_type": "subslide"}
# ## 命名實體

# %% id="BREfnaQcaQ3W"
from spacy import displacy

# %% colab={"base_uri": "https://localhost:8080/", "height": 157} id="KozpzTVNYLnb" outputId="491d9b9c-c2a5-45dd-c07e-b35c38fe3894"
text = """
長榮大學今天與勞動部台南就業中心舉辦「勇往職前･迎薪未來」校園徵才；
包括電腦資訊、餐飲旅店、文創、休閒運動、零件製造、醫療器材、
服飾用品等共34家在地廠商釋出含工讀與全職共923職缺
"""
doc = nlp(text)
displacy.render(doc, style='ent',jupyter=True)

# %% [markdown] id="veULoMBuyvmw" slideshow={"slide_type": "subslide"}
# ## 依存句法

# %% colab={"base_uri": "https://localhost:8080/", "height": 373} id="U32pNPApXs4C" outputId="c8badc9f-3f8c-4f0e-835f-eb381f083102"
text = "我想要一份2號餐"
doc = nlp(text)
displacy.render(doc, style='dep',jupyter=True, options={'distance':130})

# %% [markdown] id="eqLB6wQZJNdj" slideshow={"slide_type": "slide"}
# # 商品評論資料集 

# %% [markdown] id="uJ6IRZUcJx5H" slideshow={"slide_type": "subslide"}
# ## 載入資料集

# %% id="NVU87c0tKxPf"
import pandas as pd

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="54llyAjuJbi3" outputId="ced4346b-db28-4cee-adc6-6d01612e5655"
data_path = "https://raw.githubusercontent.com/howard-haowen/Chinese-NLP/main/online_shopping_5_cats_tra.csv"
df = pd.read_csv(data_path)
df.sample(5, random_state=900)

# %% colab={"base_uri": "https://localhost:8080/"} id="nnfwJ0aPJipM" outputId="6a573693-b16b-4c97-9511-a8dc717918e9"
df['cat'].value_counts()

# %% colab={"base_uri": "https://localhost:8080/"} id="HOxJ2lgHJjqq" outputId="3d5f4b80-212c-4356-e6b9-7206ddb9b0f9"
df['label'].value_counts()

# %% [markdown] id="G8DufoY5QH3J" slideshow={"slide_type": "subslide"}
# ## 文本預處理

# %% colab={"base_uri": "https://localhost:8080/", "height": 36} id="l7SoykhOUotP" outputId="10635afe-a957-43b8-ef15-c127e3168440"
text = df.at[31902, 'review']
text

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="fckruMO7Q1Ld" outputId="11e74c95-b201-4a51-dbbe-9805946763b7"
downsized_df = df.sample(5000, random_state=100)
downsized_df.tail()

# %% colab={"base_uri": "https://localhost:8080/"} id="k6NcmmDMRMRv" outputId="a787752e-efb5-4b35-eae9-ca90e4bf856b"
downsized_df['cat'].value_counts()

# %% colab={"base_uri": "https://localhost:8080/"} id="owdSAvEaRQGz" outputId="001f8307-aaf8-4090-f37a-4a786cf36c4e"
downsized_df['label'].value_counts()

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="JMQIRyBcRy76" outputId="940d7553-18c4-40b9-bb2b-059abeeea1e3"
downsized_df.reset_index(drop=True, inplace=True)
downsized_df.tail()

# %% id="Ql3deDTDMDE1"
from spacy.tokens import Doc

class WhitespaceTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.strip().split()
        return Doc(self.vocab, words=words)


# %% id="M7DHrR3jSu3J"
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


# %% id="9fiaLikpWIix"
import time

# %% colab={"base_uri": "https://localhost:8080/"} id="xgrN-GXsTAhS" outputId="f3f4f762-fc71-4b11-ff45-9209eea69091"
start = time.time()
downsized_df['tokens'] = downsized_df['review'].apply(preprocess_text)
end = time.time()
print(f"總耗時{end-start}秒")

# %% colab={"base_uri": "https://localhost:8080/", "height": 293} id="RaOtxEOATZ3t" outputId="73be1547-572a-48b4-f663-52afa5e0b3c1"
downsized_df.tail()

# %% [markdown] id="UWyZMhzFUOrE" slideshow={"slide_type": "subslide"}
# ## 切割資料集

# %% id="8dbgL2UJUWm1"
import numpy as np

def train_validate_test_split(df, train_percent=.6, validate_percent=.2, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return train, validate, test


# %% id="-B1AjhmpUZsx"
# train, validate, test = train_validate_test_split(downsized_df, seed=100)

# %% id="XlpenGs1d19N"
from sklearn import feature_extraction

# %% id="ezJTwMaudzid"
train, test = model_selection.train_test_split(downsized_df, test_size=0.3, random_state=100)

# %% colab={"base_uri": "https://localhost:8080/"} id="DUYDJev6UdTB" outputId="5266ca0f-8e70-4819-eefd-01b1b041bd78"
print(train['cat'].value_counts())
print(train['label'].value_counts())
print(test['cat'].value_counts())
print(test['label'].value_counts())

# %% [markdown] id="w7MHZyDCJyWM" slideshow={"slide_type": "slide"}
# # 文本表徵

# %% [markdown] id="jqbOq1ptgKmq"
#  詞頻: TF-IDF
#  * 淺層神經網路下的靜態詞向量: fasttext
#  * 深層神經網路下的動態向量: BERT (CNN + transformer)

# %% [markdown] id="_j3urOZK-eDh" slideshow={"slide_type": "subslide"}
# ## 圖片向量化
#
# ![](data:image/jpeg;base64,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)

# %% [markdown] id="ePaa3CuJhERO" slideshow={"slide_type": "subslide"}
# ## 文本向量化

# %% [markdown] id="wMdcDcD3eNCP"
#  * 象徵式表徵: TF-IDF (依據詞頻)
#  * 分佈式表徵: fasttext (靜態詞向量)
#  * 分散式表徵: transformer (動態字向量)

# %% [markdown] id="OkXUDKlaJ7Z_" slideshow={"slide_type": "subslide"}
# ### TF-IDF

# %% [markdown] id="l2u951-CQKxA"
# ![](https://miro.medium.com/max/1000/1*vWWmJlDykVRkjg9c38VbxQ.png)

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="4RzA97fLWdOn" outputId="aebc48d6-2a07-4d04-de98-75781b6b5b82"
# !pip install -U texthero

# %% colab={"base_uri": "https://localhost:8080/"} id="sIi-oXx2j5po" outputId="6adc52c3-9206-4b67-8e44-f01cd899fb01"
# !pip show spacy

# %% colab={"base_uri": "https://localhost:8080/"} id="_FsMZZsOmC5h" outputId="920ea99e-5291-4a48-bc33-b768becce6cf"
# !pip list | grep spacy

# %% id="L54vkQgVKDrX"
import texthero as hero

# %% colab={"base_uri": "https://localhost:8080/"} id="MiWOGGwoKIAV" outputId="5bdc697d-a842-49b5-af0f-e8e6ed023143"
start = time.time()
train['tfidf'] = hero.tfidf(train['tokens'])
end = time.time()
print(f"總耗時{end-start}秒")

# %% colab={"base_uri": "https://localhost:8080/", "height": 293} id="8fd8mlwIWvJz" outputId="8dd996ac-47d4-45b1-88a5-4be1aa7a2035"
train[['label', 'cat', 'tokens', 'tfidf']].tail()

# %% colab={"base_uri": "https://localhost:8080/"} id="BMxKRQ8HslkD" outputId="1268ec38-ff1d-4e22-98fa-6a7ca8017044"
len(train.iloc[0]['tfidf'])

# %% [markdown] id="pEMk2GHyLIAO" slideshow={"slide_type": "subslide"}
# ### PCA

# %% [markdown] id="FS_zJujkRv8z"
# ![](https://3.bp.blogspot.com/-aGt6JrHLt_E/WjXd4Ge7B-I/AAAAAAAAAgY/-gF8sXb4qqAwObOmP9hTiJa0ot5wR82ZACLcBGAs/s1600/Principal-Component-Analysis-And-Dimensionality-Reduction.jpg)

# %% colab={"base_uri": "https://localhost:8080/"} id="WCoeoCZsW1O9" outputId="9b35fa66-6aad-460e-d538-f42412ecc77b"
start = time.time()
train['pca_from_tfidf'] = hero.pca(train['tfidf'])
end = time.time()
print(f"總耗時{end-start}秒")

# %% colab={"base_uri": "https://localhost:8080/", "height": 293} id="5_5we3k3LIeH" outputId="44794ef2-4f1d-4a43-bd3f-6df6c045b70f"
train[['label', 'cat', 'tokens', 'pca_from_tfidf']].tail()

# %% colab={"base_uri": "https://localhost:8080/", "height": 542} id="gIvsvZilMY-R" outputId="86226df5-c484-446e-a8e4-f1bb688ae5d0"
hero.scatterplot(train, col='pca_from_tfidf', color='cat', title="商品評論訓練集PCA")

# %% colab={"base_uri": "https://localhost:8080/", "height": 542} id="Lzf2L_ZUpnYh" outputId="571554f9-33da-43f8-8b40-6e0bd22b92c9"
hero.scatterplot(train, col='pca_from_tfidf', color='label', title="商品評論訓練集PCA")

# %% colab={"base_uri": "https://localhost:8080/"} id="B8ZU0ELGM132" outputId="812d13fd-0e96-4555-f908-1b36cb7f4d2d"
NUM_TOP_WORDS = 10
train.groupby('cat')['tokens'].apply(lambda x: hero.top_words(x)[:NUM_TOP_WORDS])

# %% [markdown] id="8xImkr2Rpzoa" slideshow={"slide_type": "subslide"}
# ### word2vec

# %% [markdown] id="9VrjxDV1Ounm"
# 分佈式語意基本假設
#
# ![](https://slideplayer.com/slide/12147948/71/images/10/John+Rupert+Firth+You+shall+know+a+word+by+the+company+it+keeps.jpg)
#

# %% [markdown] id="T452ZOBJTyy-"
# - 藍色目標詞
# - 紅色語境詞
# - 自監督學習
#
# ![](https://twice22.github.io/images/word2vec/cbow-example.png)

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="HX783PuLmahX" outputId="5aee0091-5c8e-4496-e3f8-4eed2b97d431"
# !pip install -U spacy
# !python -m spacy download zh_core_web_md

# %% colab={"base_uri": "https://localhost:8080/"} id="gGhwQvr-WDQB" outputId="366b8fd8-ea68-4b6d-d523-4e097ea7aa22"
# !pip show spacy

# %% colab={"base_uri": "https://localhost:8080/"} id="52OiKCh1bVxI" outputId="6b561f97-45ec-4577-f429-19279188f19c"
# !pip install -q whatlies

# %% id="4Frv8hiGbexz"
from whatlies import EmbeddingSet
from whatlies.language import SpacyLanguage

lang = SpacyLanguage('zh_core_web_md')

# %% colab={"base_uri": "https://localhost:8080/", "height": 385} id="9-Jadg3tcIwk" outputId="485940f0-d0b1-415d-973d-de27891ba1d3"
words = ['醫生', '護士', '醫院', '教授', '學生', '學校']
emb = lang[words]
emb.plot_interactive(x_axis='醫院', y_axis='學校')

# %% [markdown] id="oOVU5VCCq1f1"
# 餘弦相似性
#
# ![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQHFjinj2_lYY-tQ9rTzOiKpwovrfzJ60QXPg&usqp=CAU)

# %% [markdown] id="9ZzgHzxmL2Ha"
# [向量投影視覺化](https://projector.tensorflow.org/)

# %% [markdown] id="mYI4Y01bOksK" slideshow={"slide_type": "subslide"}
# ### fasttext
# - 字元等級Ngram
# - 合成未登錄詞(OOV)的向量
#
# ![](https://amitness.com/images/fasttext-center-word-embedding.png)

# %% colab={"base_uri": "https://localhost:8080/"} id="Z5r-i8MNp8hH" outputId="c5403928-b6a2-4700-9530-4f56281a8f5b"
# !pip install -q fasttext

# %% id="VuRX2Bi5oojr"
import fasttext

# %% id="DIYMQ89SqCMk"
texts = downsized_df['tokens'].to_list()

with open('corpus.txt', 'w') as f:
    for item in texts:
        f.write(f"{item}\n")

# %% colab={"base_uri": "https://localhost:8080/"} id="bFgscYFEqGJM" outputId="b6822e15-570a-4a6a-fb1d-e448e0ad4e84"
start = time.time()
model = fasttext.train_unsupervised('corpus.txt')
end = time.time()
print(f"總耗時{end-start}秒")

# %% colab={"base_uri": "https://localhost:8080/"} id="YcidT-VqqJnF" outputId="fbf6484a-bae5-46ff-b4c2-4123313d66e0"
len(model.words)

# %% colab={"base_uri": "https://localhost:8080/"} id="ptBdgvpnqOFD" outputId="5d6e9696-0c0f-427e-d000-3813142f2655"
model.words[:20]

# %% colab={"base_uri": "https://localhost:8080/"} id="5tNi1mOSqTTo" outputId="c4772673-0255-4d88-cee7-b5ff36e92184"
model.get_nearest_neighbors('好評')

# %% colab={"base_uri": "https://localhost:8080/"} id="ZQy9_8J-qbcZ" outputId="60ac455b-7eaa-4c5d-c856-58b88f810a24"
model.get_nearest_neighbors('差評')

# %% [markdown] id="GSMkhmy0O7sD" slideshow={"slide_type": "subslide"}
# ### transformer
#
# - GPT
# - BERT
#
# ![](https://raw.githubusercontent.com/jessevig/bertviz/master/images/head-view.gif)

# %% [markdown] id="2WQmsVkltTfc" slideshow={"slide_type": "slide"}
# # 遷移學習
#
# ![](https://ruder.io/content/images/2019/08/pretraining_adaptation.png)

# %% [markdown] id="qzTKcH1z1R6D"
# ## 遷移學習的類別
#
# ![](https://ruder.io/content/images/2019/08/transfer_learning_taxonomy.png)

# %% [markdown] id="Cj8uYd4JqEjS"
# ## spaCy的遷移學習架構
#
# ![](https://spacy.io/tok2vec-listener-8c4d53807708b270c07a085f4a2da75f.svg)

# %% [markdown] id="iPEOKDWxlfK8" slideshow={"slide_type": "slide"}
# # 文本表徵的具體應用

# %% [markdown] id="KnPkbsmIOWtx" slideshow={"slide_type": "subslide"}
# ## 文本相似性

# %% id="GXcHHAGOgp-A"
nlp = spacy.load("zh_core_web_md")

# %% colab={"base_uri": "https://localhost:8080/"} id="__vUOwVVf9D2" outputId="4b2e1263-a6ef-4d3c-b2b1-104d7d1fbd05"
doc01 = nlp("防Omicron威脅 一至三類今起可打第3劑")
doc02 = nlp("第3劑開打…賣場設站隨到隨打 首波4.8萬人符資格")
doc03 = nlp("西半部低溫特報 全台平地最冷在基隆僅8度")

sim01 = doc01.similarity(doc02)
sim02 = doc01.similarity(doc03)

print(sim01)
print(sim02)

# %% colab={"base_uri": "https://localhost:8080/"} id="AQrx7oKAMPon" outputId="2b966a7c-3d90-4707-abcc-3ca7c78120e1"
# !pip install -q faiss-cpu

# %% id="_eVx5XTvMSJZ"
import faiss

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


# %% id="4jWRvSNtMjnn"
text_list = downsized_df['review'].to_list()
index_list = downsized_df.index.values

# %% colab={"base_uri": "https://localhost:8080/"} id="dYv6bsmAMY_v" outputId="fc0a67f3-061f-4b11-88fa-dff6ff5cc505"
start = time.time()
doc_vectors = []
for doc in nlp.pipe(text_list):
    doc_vectors.append(doc.vector)
end = time.time()
print(f"總耗時{end-start}秒")

# %% colab={"base_uri": "https://localhost:8080/"} id="v61M0Il801ok" outputId="94603fdc-9aef-4127-c592-191076e490bf"
start = time.time()
index, embeddings = create_index_embeddings(doc_vectors, index_list)
end = time.time()
print(f"總耗時{end-start}秒")

# %% colab={"base_uri": "https://localhost:8080/"} id="xagAoKlNEzXi" outputId="46c50e23-4129-4062-bbf9-b279317cde0f"
text_list[-3:]

# %% colab={"base_uri": "https://localhost:8080/"} id="Z9Kpa49Y0pvp" outputId="1b022d3f-f306-43c3-a2d2-d2dadfb38275"
index_list[-3:]

# %% id="xZ6lD1QFQGYG"
D, I = index.search(np.array([embeddings[4999]]), k=5)
SNs = I.flatten().tolist()

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="2nBfqs1MQiD9" outputId="eca3b14f-9286-42f5-c04e-1379a294996e"
downsized_df.loc[SNs, ["review", "cat", "label"]]

# %% [markdown] id="-HD7iSDvjK2f" slideshow={"slide_type": "subslide"}
# ## 文本分類

# %% [markdown] id="sja1AIgm104E"
# ### 使用TF-IDF

# %% colab={"base_uri": "https://localhost:8080/", "height": 293} id="bOSQmqz094w7" outputId="4efca8cf-6b20-4b0e-e185-520a935e3ff7"
downsized_df.tail()

# %% id="9dtULrFQ92Cd"
from sklearn import feature_extraction, model_selection, naive_bayes, pipeline

# %% id="5TchZgMy93rN"
train_df, test_df = model_selection.train_test_split(downsized_df, test_size=0.3, random_state=100)
train_y = train_df["cat"].values
test_y = test_df["cat"].values

# %% id="92ySI6uf-sAv"
vectorizer = feature_extraction.text.TfidfVectorizer(max_features=10000, ngram_range=(1,2))
corpus = train_df["tokens"]
vectorizer.fit(corpus)
train_X = vectorizer.transform(corpus)

# %% id="Q_WLJpwqBOvs"
classifier = naive_bayes.MultinomialNB()
model = pipeline.Pipeline([("vectorizer", vectorizer),  
                           ("classifier", classifier)])

# %% colab={"base_uri": "https://localhost:8080/"} id="N92oJ_ZuYhaF" outputId="c87af2c6-cc56-461b-9d5d-5b18dc56246f"
start = time.time()
model["classifier"].fit(train_X, train_y)
end = time.time()
print(f"總耗時{end-start}秒")

# %% colab={"base_uri": "https://localhost:8080/"} id="lTj2-RXSBfIq" outputId="0790ccef-c3f9-45f9-a14f-82d672616b49"
test_X = test_df["tokens"].values
start = time.time()
predicted = model.predict(test_X)
end = time.time()
print(f"總耗時{end-start}秒")

# %% id="Nqfs2AdIEt0c"
from sklearn import metrics

# %% colab={"base_uri": "https://localhost:8080/", "height": 300} id="vYRPpKwFBuH_" outputId="87cda613-6c29-4dff-f99b-6ab43d640d91"
class_report = metrics.classification_report(test_y, predicted, output_dict=True)
report_df = pd.DataFrame(class_report).transpose()
report_df

# %% id="Wy_cDfEXbR_a"
from sklearn.svm import SVC 

svc_clf = SVC(C=1, gamma="auto", kernel='linear',probability=False)
model = pipeline.Pipeline([("vectorizer", vectorizer),  
                           ("svm", svc_clf)])

# %% colab={"base_uri": "https://localhost:8080/"} id="5tyKdUKVas-8" outputId="e5d0bd94-983a-4b83-970e-fdfb47899954"
start = time.time()
model["svm"].fit(train_X, train_y)
end = time.time()
print(f"總耗時{end-start}秒") 

# %% colab={"base_uri": "https://localhost:8080/"} id="kp1e-cXGcCoM" outputId="c6b78413-f860-495f-e505-21307bde387c"
start = time.time()
predicted = model.predict(test_X)
end = time.time()

print(f"總耗時{end-start}秒")

# %% colab={"base_uri": "https://localhost:8080/", "height": 300} id="5oZYxDHAceyZ" outputId="449fd6b8-a047-4c77-c138-021247613a11"
class_report = metrics.classification_report(test_y, predicted, output_dict=True)
report_df = pd.DataFrame(class_report).transpose()
report_df

# %% [markdown] id="wZtjuVRw17S-"
# ### 使用fasttext

# %% colab={"base_uri": "https://localhost:8080/"} id="NbRP2S8YIZvY" outputId="b7777ea4-1d8b-4632-89a8-cd62f3205358"
all_texts = train_df['tokens'].tolist()
all_labels = train_df['cat'].tolist()
prep_datapoints=[]
for i in range(len(all_texts)):
    sample = '__label__'+ str(all_labels[i]) + ' '+ all_texts[i]
    prep_datapoints.append(sample)
prep_datapoints[:3]

# %% id="f2Wtm3gXJJ9I"
with open('train_fasttext.txt','w') as f:
    for datapoint in prep_datapoints:
        f.write(datapoint)
        f.write('\n')
    f.close()

# %% colab={"base_uri": "https://localhost:8080/"} id="gK5XpySYJQWA" outputId="81335c10-931a-4a72-e9fb-2e89770c11c7"
start = time.time()
model = fasttext.train_supervised('train_fasttext.txt')
end = time.time()
print(f"總耗時{end-start}秒")

# %% colab={"base_uri": "https://localhost:8080/", "height": 36} id="izxql8OOKgkX" outputId="e983cde4-13e5-4517-efe6-9d76475836ad"
sample = test_df.iloc[33]['tokens']
sample

# %% colab={"base_uri": "https://localhost:8080/", "height": 36} id="CwwWQNWHKuQi" outputId="baad77ac-d88d-429c-e66d-f83b66da031d"
answer = test_df.iloc[33]['cat']
answer

# %% colab={"base_uri": "https://localhost:8080/"} id="36P5PKPwK0US" outputId="eb7cd61c-9e64-4778-b3b3-1f8ed7b2e813"
predicted = model.predict(sample)
predicted

# %% colab={"base_uri": "https://localhost:8080/"} id="HWPV4zPLJ7Fm" outputId="e7d43792-3dce-49dc-a875-38e7a9b530d2"
all_texts = test_df['tokens'].tolist()
all_labels = test_df['cat'].tolist()
prep_datapoints=[]
for i in range(len(all_texts)):
    sample = '__label__'+ str(all_labels[i]) + ' '+ all_texts[i]
    prep_datapoints.append(sample)
prep_datapoints[:3]

# %% id="u2usR6PbKKXi"
with open('test_fasttext.txt','w') as f:
    for datapoint in prep_datapoints:
        f.write(datapoint)
        f.write('\n')
    f.close()

# %% colab={"base_uri": "https://localhost:8080/"} id="o45eN8nNKR53" outputId="ffc8f657-8256-47e5-bfc6-9d0d08be1e58"
model.test("test_fasttext.txt")

# %% colab={"base_uri": "https://localhost:8080/"} id="kFL_VI5gfeDD" outputId="c85e7332-e427-4a1a-bac2-200f9d6a0aaa"
predicted = [model.predict(text)[0][0].split("__")[-1] for text in test_X] 
predicted[-3:]

# %% colab={"base_uri": "https://localhost:8080/"} id="Ais8AZ7JgdHd" outputId="66c253a6-33dc-44f9-f607-8097bfdffb6f"
test_X[-3:]

# %% colab={"base_uri": "https://localhost:8080/", "height": 300} id="XmutUcpUeFlN" outputId="dabefea9-e06d-4058-a627-7204b97f0d48"
class_report = metrics.classification_report(test_y, predicted, output_dict=True)
report_df = pd.DataFrame(class_report).transpose()
report_df

# %% [markdown] id="3V9F5-Zr2AG1"
# ### 使用transformer

# %% colab={"base_uri": "https://localhost:8080/"} id="Lf4BpyZZl87h" outputId="edbcb014-11b5-4db0-e302-28a32104473c"
# !nvcc --version

# %% colab={"base_uri": "https://localhost:8080/"} id="nk2zMpIGlr2u" outputId="23b785d4-ca19-4b34-e8ac-e392b5f791b6"
# !pip install -U pip setuptools wheel
# !pip install -U spacy[cuda111,transformers]

# %% colab={"base_uri": "https://localhost:8080/"} id="OZnUHGTkj_NG" outputId="da647db3-7b22-4da0-a789-aaf8ac108bd8"
# !python -m spacy download zh_core_web_trf

# %% id="cQXwZrhRkEwa"
import spacy

spacy.prefer_gpu()
nlp = spacy.load("zh_core_web_trf")

# %% colab={"base_uri": "https://localhost:8080/"} id="4QJIEEYerDAS" outputId="9902160c-e2ec-469e-f607-36e3fb9adf1a"
nlp.pipe_names

# %% colab={"base_uri": "https://localhost:8080/"} id="ibj9mtCDn6nS" outputId="eb61f0f8-4e5f-4428-b030-54a2923999c6"
text = "送人的，不知道好不好。聽說紅心火龍果還是比較甜的。"
doc = nlp(text)
doc._.trf_data.wordpieces

# %% colab={"base_uri": "https://localhost:8080/"} id="zMu2zjbGoucy" outputId="47f05931-670c-428e-bf27-4324425658d8"
doc._.trf_data.tensors[0].shape

# %% colab={"base_uri": "https://localhost:8080/"} id="vTeI9wK_sCZc" outputId="a88aaf19-755c-46ec-8992-84e0d01c7ec4"
text_labels = train_df['cat'].unique().tolist()
text_labels

# %% colab={"base_uri": "https://localhost:8080/"} id="80ITsjJJo_AE" outputId="3a92a912-f755-4a71-ad84-1aeb2f773ed1"
train_df['tuples'] = train_df.apply(lambda row: (row['review'], row['cat']), axis=1)
train_data = train_df['tuples'].tolist()
test_df['tuples'] = test_df.apply(lambda row: (row['review'], row['cat']), axis=1)
test_data = test_df['tuples'].tolist()

# %% colab={"base_uri": "https://localhost:8080/"} id="dS7r1bpyutNH" outputId="8e155e9d-f5c8-4ce0-b10f-2a2be29b429a"
train_data[100]

# %% id="iM7fhjwDu6bM"
from tqdm.auto import tqdm
from spacy.tokens import DocBin

def make_docs(data_list, unique_labels, dest_path):
    docs = []
    for doc, label in tqdm(nlp.pipe(data_list, as_tuples=True), total=len(data_list)):
        label_dict = {label: False for label in unique_labels}
        doc.cats = label_dict
        doc.cats[label] = True
        docs.append(doc)
    doc_bin = DocBin(docs=docs)
    doc_bin.to_disk(dest_path)



# %% colab={"base_uri": "https://localhost:8080/", "height": 81, "referenced_widgets": ["32fe9de1c45240d3bc3ca25e9bb7bb86", "8a375859a7a046e2821bc22916ac5246", "fc36d7b6c67b4642953d7e6d3bb70320", "a54510f248d74814a40624118e9e23e6", "670669e227c1424ea1967a11821196d6", "e01d27fe11764182884481768a4cb68d", "4c85e46561c940dfa1b5b6a8b0a95b8b", "13d8decc44f74da582bf1ce3521721f4", "a5baff0f9fd44801b244fbc2e01d0964", "ac15f2fff49846cba49cee0c41967ea3", "a9c73838a82e4ac78c61663607e2aa40", "afab9b71d9554783b5a0dcfe4f8d23d2", "24f68c720a4c43ceac159ea24506df93", "5342d84e474f41a79b7d9c5d3e886015", "5c08ba0fa182413aa1d11f0af7ec47ae", "c77b60a1d79e4e94bf4f230787362e81", "0eb1503563a245b6bd2d1360af2a612f", "2e80c65b32d6434fb1cd6d0628fade2a", "b8afde1754974cfe9ac3f3482b983a07", "c78c35f0e8444b3dbfa4511da23baaba", "d09dd27b15a146d4835dd25c24d09346", "c48faba8bcc64746a08b2a35a2af25c7"]} id="kVDF0GYjwTNE" outputId="82ab9d57-6c53-4b05-affb-3fd848f7258c"
make_docs(train_data, text_labels, "train.spacy")
make_docs(test_data, text_labels, "test.spacy")

# %% [markdown] id="4JObXZKa1EUL"
#  取得基本[設定檔案](https://spacy.io/usage/training)

# %% colab={"base_uri": "https://localhost:8080/"} id="Ky4x-trNw-qi" outputId="d6cde89d-c00c-4bf5-a473-f29fc4b717ed"
# %%writefile base_config.cfg

# This is an auto-generated partial config. To use it with 'spacy train'
# you can run spacy init fill-config to auto-fill all default settings:
# python -m spacy init fill-config ./base_config.cfg ./config.cfg
[paths]
train = null
dev = null

[system]
gpu_allocator = "pytorch"

[nlp]
lang = "zh"
pipeline = ["transformer","textcat"]
batch_size = 128

[components]

[components.transformer]
factory = "transformer"

[components.transformer.model]
@architectures = "spacy-transformers.TransformerModel.v3"
name = "bert-base-chinese"
tokenizer_config = {"use_fast": true}

[components.transformer.model.get_spans]
@span_getters = "spacy-transformers.strided_spans.v1"
window = 128
stride = 96

[components.textcat]
factory = "textcat"

[components.textcat.model]
@architectures = "spacy.TextCatEnsemble.v2"
nO = null

[components.textcat.model.tok2vec]
@architectures = "spacy-transformers.TransformerListener.v1"
grad_factor = 1.0

[components.textcat.model.tok2vec.pooling]
@layers = "reduce_mean.v1"

[components.textcat.model.linear_model]
@architectures = "spacy.TextCatBOW.v2"
exclusive_classes = true
ngram_size = 1
no_output_layer = false

[corpora]

[corpora.train]
@readers = "spacy.Corpus.v1"
path = ${paths.train}
max_length = 0

[corpora.dev]
@readers = "spacy.Corpus.v1"
path = ${paths.dev}
max_length = 0

[training]
accumulate_gradient = 3
dev_corpus = "corpora.dev"
train_corpus = "corpora.train"

[training.optimizer]
@optimizers = "Adam.v1"

[training.optimizer.learn_rate]
@schedules = "warmup_linear.v1"
warmup_steps = 250
total_steps = 20000
initial_rate = 5e-5

[training.batcher]
@batchers = "spacy.batch_by_padded.v1"
discard_oversize = true
size = 2000
buffer = 256

[initialize]
vectors = ${paths.vectors}

# %% colab={"base_uri": "https://localhost:8080/"} id="R91kFJ8ax7di" outputId="bb8bcad3-c833-4b4d-c277-6be63e03815c"
# !python -m spacy init fill-config ./base_config.cfg ./config.cfg

# %% id="bb0OjTdSyODb"
# !mkdir output

# %% id="I3gw3LKpysfD"
import time

# %% colab={"base_uri": "https://localhost:8080/"} id="q7xchz0yxj2u" outputId="b32f4981-d73b-45e4-80ec-2c8c970970d3"
start = time.time()
# !python -m spacy train /content/config.cfg --output /content/output --paths.train /content/train.spacy --paths.dev /content/test.spacy --gpu-id 0 --verbose
end = time.time()
print(f"總耗時{end-start}秒")

# %% colab={"base_uri": "https://localhost:8080/"} id="GHUvvCH4NxRH" outputId="15d4eb8f-058c-4d85-b454-e41e8ee35c6c"
sample_text = test_df.loc[350]['review']
sample_cat = test_df.loc[350]['cat']
print(sample_text, sample_cat)

# %% colab={"base_uri": "https://localhost:8080/"} id="o90PCYO60bKb" outputId="007aac8d-2448-47e0-e0ea-bcec498f4cd6"
nlp = spacy.load("/content/output/model-last")
doc = nlp(sample_text)
print(doc.cats)

# %% colab={"base_uri": "https://localhost:8080/", "height": 35} id="-0Tn_U1IOty5" outputId="e808ecf5-d6d0-4465-8dc3-03d3fd0c3394"
max_cat = max(doc.cats, key=doc.cats.get)
max_cat

# %% colab={"base_uri": "https://localhost:8080/"} id="S8Sj3gf8PPQy" outputId="a4d8b8e8-49f5-4174-e5b4-68017400128d"
import json

model_meta_path = "/content/output/model-best/meta.json"
with open(model_meta_path) as json_file:
    metrics = json.load(json_file)
metrics['performance'] 

# %% id="7NMeFWm5IU34"
import random

def show_test():
    idx = random.choice(test_df.index)
    text = df.at[idx, 'review']
    cat = df.at[idx, 'cat']
    predicted_proba = nlp(text).cats
    predicted_cat = max(predicted_proba, key=predicted_proba.get)
    print(f"編號:{idx}")
    print(f"評論:{text}")
    print(f"正確商品類別:{cat}")
    print(f"類別機率分佈:{json.dumps(predicted_proba, indent=4, ensure_ascii=False)}")
    print(f"模型判斷類別:{predicted_cat}")


# %% colab={"base_uri": "https://localhost:8080/"} id="UhMaxlewJ41G" outputId="1ed7e5af-5f56-4001-ffeb-2d61853d0052"
show_test()

# %% colab={"base_uri": "https://localhost:8080/"} id="GeFotoLXTDaH" outputId="b8757643-1287-4703-b291-5f85b43b381b"
show_test()

# %% [markdown] id="a11-DzEqjaJf" slideshow={"slide_type": "slide"}
# # 相關連結
#
# - [Symbolic, Distributed, and Distributional Representations for Natural Language Processing in the Era of Deep Learning: A Survey](https://www.frontiersin.org/articles/10.3389/frobt.2019.00153/full)
# - [Transfer Learning for Natural Language Processing](https://www.manning.com/books/transfer-learning-for-natural-language-processing)
# - [Mastering spaCy](https://www.packtpub.com/product/mastering-spacy/9781800563353)
