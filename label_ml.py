import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

def get_label(feature_text_str) -> str:
    
    ds = pd.read_csv("https://pdmlflasksacc.blob.core.windows.net/mldata/tax_sentences_labels.csv")
    print(f"[Dataset shape] : {ds.shape}")

    tfid = TfidfVectorizer()
    x = tfid.fit_transform(ds['Sentence'])
    le = LabelEncoder()
    y = le.fit_transform(ds['Label'])

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)

    model = LogisticRegression()
    model.fit(x_train, y_train)

    y_train_predict = model.predict(x_train)
    acc  = accuracy_score(y_train,y_train_predict)
    print(f"[Accuracy] : {acc}")

    test_feature = [feature_text_str]
    test_feature_trns = tfid.transform(test_feature)
    prediction = model.predict(test_feature_trns)
    prediction_label = le.inverse_transform(prediction)
    print(f"[Prediction] : {prediction_label}")

    return prediction_label[0]

