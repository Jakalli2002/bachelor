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
    takes in a text, preprocesses it and retruns the preprocessed lemmatization of that text
    """
    collector = []
    for sentence in nltk.sent_tokenize(text, language="german"): # turns text into sentence tokens
        tokens = nltk.word_tokenize(sentence, language="german") # turns sentence tokens into word tokens
        for _, lemma, _ in tagger.tag_sent(tokens):
            if lemma.lower() in german_stopwords:
                continue
            if not re.match(r"^\w+$", lemma):
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
    Turns the lemmatized texts into TF-IDF vectors.
    """
    vectorizer = TfidfVectorizer(min_df=MIN_DF, max_features=MAX_FEATURES)
    X_train = vectorizer.fit_transform(train_lemmas)
    X_test = vectorizer.transform(test_lemmas)
    return X_train, X_test

def run_classic(train_lemmas, train_labels, test_lemmas):
    """
    runs the logistic regression model from sklearn (vectorize -> train -> predict). returns the predicted categories for the test set.
    """
    X_train, X_test = vectorize(train_lemmas, test_lemmas)

    model = LogisticRegression(C=C_VALUE, max_iter=MAX_ITER)
    model.fit(X_train, train_labels)

    return model.predict(X_test)


# -------------------- TESTING --------------------
# lemmatizing
"""
if __name__ == "__main__":
    import time
    from data import load_data
    from config import TRAINING_PATH

    train_df = load_data(TRAINING_PATH)
    artikel_liste = train_df["text"].head(20).tolist()

    start = time.perf_counter()
    for a in artikel_liste:
        preprocessing(a)
    dauer = time.perf_counter() - start

    print(f"{dauer/20:.3f}s pro Artikel  →  {dauer/20 * 10273 / 60:.1f} min gesamt")

    print("\nORIGINAL:\n", artikel_liste[0][:300])
    print("\nLEMMATISIERT:\n", preprocessing(artikel_liste[0])[:300])
"""

# save/load df
"""
if __name__ == "__main__":
    from data import load_data
    from config import TRAINING_PATH, TEST_PATH, TRAIN_LEMMAS_PATH, TEST_LEMMAS_PATH

    train_df = get_preprocessed(load_data(TRAINING_PATH), TRAIN_LEMMAS_PATH)
    test_df = get_preprocessed(load_data(TEST_PATH), TEST_LEMMAS_PATH)

    print(train_df.shape, test_df.shape)
    print(train_df.columns.tolist())
    print("\nORIGINAL:\n", train_df["text"].iloc[0][:200])
    print("\nLEMMAS:\n", train_df["lemmas"].iloc[0][:200])
"""

# model
if __name__ == "__main__":
    from data import load_data, get_data_subset
    from config import TRAINING_PATH, TEST_PATH, TRAIN_LEMMAS_PATH, TEST_LEMMAS_PATH

    train_df = get_preprocessed(load_data(TRAINING_PATH), TRAIN_LEMMAS_PATH)
    test_df = get_preprocessed(load_data(TEST_PATH), TEST_LEMMAS_PATH)

    subset = get_data_subset(train_df, 1.0, 1)
    preds = run_classic(subset["lemmas"], subset["category"], test_df["lemmas"])

    print("Trainingsartikel:", len(subset))
    print("Accuracy:", (preds == test_df["category"]).mean())
    print("Erste Vorhersagen:", preds[:10])