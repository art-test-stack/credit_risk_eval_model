from utils import GLOVE_MODEL

import stanza
import spacy
import numpy as np
import torch
import pandas as pd

from typing import Tuple, List

import nltk
from nltk.corpus import stopwords
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


class StanfordNLP:
    def __init__(
            self, 
            emb_dim: int = 200,
            stop_words: bool = False,
            verbose: bool = True
        ) -> None:
        # nltk.download('stopwords')
        # stanza.download('en')
        self.stop_words = set(stopwords.words('english'))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.corenlp = stanza.Pipeline(
            'en', 
            processors='tokenize,pos,lemma,depparse,ner',
            use_gpu=torch.cuda.is_available(),
            device=self.device,
        )
        self.verbose = verbose
        self.use_sw = stop_words
        # self.glove = spacy.load('en_core_web_lg')  # or 'en_vectors_web_lg' for larger GloVe vectors

        self.glove = self.load_spacy_glove(GLOVE_MODEL)
        self.embedding_dim = emb_dim

    def load_spacy_glove(self, glove_file_path: str) -> spacy.language.Language:
        nlp = spacy.blank("en")
        with open(glove_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                nlp.vocab.set_vector(word, vector)
        return nlp

    def tokenize(self, text: str) -> List[str]:
        doc = self.corenlp(text)
        if self.use_sw:
            tokens = [word.lemma for sentence in doc.sentences for word in sentence.words if word.lemma not in self.stop_words]
        else:
            tokens = [word.lemma for sentence in doc.sentences for word in sentence.words]
        return tokens
    
    def pad_embeddings(self, embeddings: np.ndarray, max_length: int = 200) -> np.ndarray:
        if len(embeddings) > max_length:
            return embeddings[:max_length]
        else:
            padding = np.zeros((max_length - len(embeddings), embeddings.shape[1]))
            return np.vstack((embeddings, padding))

    def embbed_tokens(self, tokens: List[str]) -> np.ndarray:
        emb = np.array([ self.glove.vocab.get_vector(tok) for tok in tokens ])
        emb = self.pad_embeddings(emb, 200)
        return emb
    
    def __call__(self, text: str) -> Tuple[List[str], np.ndarray]:
        tokens = self.tokenize(text)
        emb = self.embbed_tokens(tokens)
        return emb
    
    def process_batch(
            self, 
            texts: List[str] | pd.Series | pd.DataFrame, 
            max_workers: int | None = None, 
            parralize: bool = True
        ) -> np.ndarray:
        if isinstance(texts, pd.Series):
            texts = texts.values
        elif isinstance(texts, pd.DataFrame):
            texts = texts["desc"].values

        if self.device == "cuda" and parralize:
            res = self.process_batch_cuda(texts, max_workers)
        elif self.device == "cpu" and parralize:
            res = self.process_batch_cpu(texts, max_workers)
        else:
            res = np.array([ self(text) for text in tqdm(texts)])
        return res
    
    def process_batch_cuda(self, texts: List[str], max_workers: int | None) -> np.ndarray:
        res = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for emb in tqdm(
                executor.map(self, texts), 
                total=len(texts), 
                colour="blue",
                disable=not self.verbose
            ):
                res.append(emb)
        return np.array(res)
    
    def process_batch_cpu(self, texts: List[str], max_workers: int | None) -> np.ndarray:
        res = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for emb in tqdm(
                executor.map(self, texts), 
                total=len(texts), 
                colour="magenta",
                disable=not self.verbose
            ):
                res.append(emb)
        return res