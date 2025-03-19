import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag, DependencyGraph
import gensim.downloader as api
import os

# nltk_data_path = os.path.expanduser("~") + "/nltk_data"  # Store in home directory
# os.makedirs(nltk_data_path, exist_ok=True)
# nltk.data.path.append(nltk_data_path)
nltk.download('punkt_tab')
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")
nltk.download('averaged_perceptron_tagger_eng')

# Load word embeddings (GloVe 50D)
embeddings = api.load("glove-wiki-gigaword-50")

def Sentence_Segmentation(text):
    return sent_tokenize(text)

def Word_Tokenization(text):
    return word_tokenize(text)

def Stemming(words):
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in words]

def Lemmatization(words):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in words]

def Stop_Word_Analysis(words):
    stop_words = set(stopwords.words("english"))
    return [word for word in words if word.lower() not in stop_words]

def POS_Tagging(words):
    return pos_tag(words)

def Dependency_Parsing(pos_tags):
    return [(word, tag) for word, tag in pos_tags]

def Word_Embeddings(words):
    return {word: embeddings[word].tolist() for word in words if word in embeddings}


st.title("NLP Pipeline using NLTK")
text_input = st.text_area("Enter text for processing:")

if st.button("Process Text"):
    if text_input:
        ss = Sentence_Segmentation(text_input)
        st.subheader("Sentence Segmentation")
        st.write(ss)

        wt = Word_Tokenization(text_input)
        st.subheader("Word Tokenization")
        st.write(wt)

        st_words = Stemming(wt)
        st.subheader("Stemming")
        st.write(st_words)

        lm_words = Lemmatization(wt)
        st.subheader("Lemmatization")
        st.write(lm_words)

        swa = Stop_Word_Analysis(wt) 
        st.subheader("Stop Word Analysis")
        st.write(swa)

        pos = POS_Tagging(wt)
        st.subheader("Part-of-Speech (POS) Tagging")
        st.write(pos)

        dp = Dependency_Parsing(pos)        
        st.subheader("Dependency Parsing")
        st.write(dp)
        
        we = Word_Embeddings(wt)                                
        st.subheader("Word Embeddings")
        st.write(we)
    else:
        st.warning("Please enter some text.")
