"""
classic approach: preprocessing and lemmatization (one time, cached in processed folder) -> vectorization (tf-idf) and model run (logistic regression)
"""
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from HanTa import HanoverTagger as ht
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from config import MIN_DF, MAX_FEATURES, C_VALUE, MAX_ITER


# -------------------- PRE LOADING --------------------
tagger = ht.HanoverTagger("morphmodel_ger.pgz")
german_stopwords = set(stopwords.words("german"))


# -------------------- PREPROCESSING --------------------
def preprocessing(text: str) -> str:
    """
    takes in a text, preprocesses it and returns the lemmatized text as a string.

    steps: split into sentences (nltk) -> tokenize each sentence (nltk) ->
    tag and lemmatize with HanTa -> drop stopwords and non-words -> join lemmas.

    tagging is done per sentence because HanTa uses the sentence context to
    determine the part of speech, which the lemma depends on.
    no lowercasing here: capitalization is a signal for nouns in german and helps
    the tagger. lowercasing happens later in the TfidfVectorizer (lowercase=True).
    """
    collector = []
    for sentence in nltk.sent_tokenize(text, language="german"): # turns text into sentence tokens
        tokens = nltk.word_tokenize(sentence, language="german") # turns sentence tokens into word tokens
        for _, lemma, _ in tagger.tag_sent(tokens):
            if lemma.lower() in german_stopwords: # filters stopwords
                continue
            if not re.match(r"^\w+$", lemma): # filters punctuation marks
                continue
            collector.append(lemma)

    return " ".join(collector)

# preprocessing for the whole dataframe
def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    does the preprocessing for the whole dataframe and adds a new columm (lemma) so the df has the category, original text and lemma.
    the original text is for the trainng of BERT and lemma for the logistic regression
    """
    df = df.copy()
    tqdm.pandas(desc="Lemmatisierung")
    df["lemmas"] = df["text"].progress_apply(preprocessing)
    return df

# saving/loading the df so preprocessing has to be done just once
def get_preprocessed(df: pd.DataFrame, cache_path) -> pd.DataFrame:
    """
    saves/loads the preprocessed df. cache file has to be deleted if the preprocessing changes or it will load the old one
    """
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    result = preprocess_df(df)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(cache_path)
    return result


# -------------------- VECTORIZE AND MODEL --------------------
def vectorize(train_lemmas, test_lemmas):
    """
    turns the lemmatized texts into TF-IDF vectors.
    the vocabulary and idf weights are fitted on the training subset only, so the
    vocabulary size depends on the amount of training data (which is what the
    learning curve measures) and no test data leaks into the model.
    """
    vectorizer = TfidfVectorizer(min_df=MIN_DF, max_features=MAX_FEATURES, lowercase=True)
    X_train = vectorizer.fit_transform(train_lemmas)
    X_test = vectorizer.transform(test_lemmas)
    return X_train, X_test

def run_classic(train_lemmas, train_labels, test_lemmas):
    """
    runs the logistic regression model from sklearn (vectorize -> train -> predict). returns the predicted categories for the test set.
    metrics are calculated in experiment.py. with 9 categories sklearn automatically uses multinomial logistic regression (softmax).
    """
    X_train, X_test = vectorize(train_lemmas, test_lemmas)

    model = LogisticRegression(C=C_VALUE, max_iter=MAX_ITER)
    model.fit(X_train, train_labels)

    return model.predict(X_test)


# -------------------- TESTING --------------------
if __name__ == "__main__":
    import time
    from data import load_data, get_data_subset
    from config import TRAINING_PATH, TEST_PATH, TRAIN_LEMMAS_PATH, TEST_LEMMAS_PATH

    train_df = get_preprocessed(load_data(TRAINING_PATH), TRAIN_LEMMAS_PATH)
    test_df = get_preprocessed(load_data(TEST_PATH), TEST_LEMMAS_PATH)

    # dataframes
    #print(train_df.shape, test_df.shape)
    #print(train_df.columns.tolist())
    #print("\nORIGINAL:\n", train_df["text"].iloc[0][:200])
    #print("\nLEMMAS:\n", train_df["lemmas"].iloc[0][:200])

    # model run
    subset = get_data_subset(train_df, 0.1, 1)
    start = time.perf_counter()
    preds = run_classic(subset["lemmas"], subset["category"], test_df["lemmas"])
    dauer = time.perf_counter() - start

    print("\nTrainingsartikel:", len(subset))
    print("Accuracy:", (preds == test_df["category"]).mean())
    print(f"Dauer: {dauer:.2f}s")

    # without preprocessing, for comparison
    #preds_raw = run_classic(subset["text"], subset["category"], test_df["text"])
    #print("Ohne Preprocessing:", (preds_raw == test_df["category"]).mean())