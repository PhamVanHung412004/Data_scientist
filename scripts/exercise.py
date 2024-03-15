import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler, SMOTEN
import re


def my_func(loc):
    result = re.findall("\ [A-Z]{2}$", loc)
    if len(result) == 1:
        return result[0][1:]
    else:
        return loc


data = pd.read_excel("final_project.ods", engine="odf", dtype=str)
data["location"] = data["location"].apply(my_func)
target = "career_level"
x = data.drop(target, axis=1)
y = data[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
ros = SMOTEN(random_state=42, k_neighbors=2, sampling_strategy={"director_business_unit_leader": 500, "specialist": 500,
                                                   "managing_director_small_medium_company": 500})
print(y_train.value_counts())
print("--------------")
x_train, y_train = ros.fit_resample(x_train, y_train)
print(y_train.value_counts())

preprocessor = ColumnTransformer(transformers=[
    ("title", TfidfVectorizer(stop_words=["english"], ngram_range=(1, 1)), "title"),
    ("location", OneHotEncoder(handle_unknown="ignore"), ["location"]),
    ("description", TfidfVectorizer(stop_words=["english"], ngram_range=(1, 1), min_df=0.01, max_df=0.95),
     "description"),
    ("function", OneHotEncoder(handle_unknown="ignore"), ["function"]),
    ("industry", TfidfVectorizer(stop_words=["english"], ngram_range=(1, 1)), "industry"),
])

cls = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestClassifier()),
])

cls.fit(x_train, y_train)
y_predict = cls.predict(x_test)
print(classification_report(y_test, y_predict))
