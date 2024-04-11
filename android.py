import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
from bs4 import BeautifulSoup
import re
import string
import unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
#import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


def load_data(path):
    data = pd.read_csv(path, encoding="ISO-8859-1")
    return data

def data_info(data):
    print(data.shape)
    print(data.head(5))
    print(data.info())
    print(data.describe())
    print(data.columns)
    print(data.isnull().sum())

    pd.set_option('max_columns', None)
    for col in data.columns:
        if data[col].isna().sum() > 0:
            print(col, data[col].isna().sum())

    for col in data.columns:
        if data[col].dtypes == 'object':
            print(col, len(data[col].unique()))

    obs = len(data)
    mal = len(data.loc[data['Class'] == 1])
    not_mal = len(data.loc[data['Class'] == 0])
    print('Percentages of malware and benign applications in the original dataset:')
    print('Num of Malware: {0} ({1:.2f}%)'.format(mal, (mal / obs) * 100))
    print('Num of benign: {0} ({1:.2f}%)'.format(not_mal, (not_mal / obs) * 100))

mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", 
           "'cause": "because", "could've": "could have", "couldn't": "could not", 
           "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", 
           "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", 
           "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", 
           "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", 
           "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", 
           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have",
           "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", 
           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have",
           "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", 
           "might've": "might have","mightn't": "might not","mightn't've": "might not have", 
           "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", 
           "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", 
           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", 
           "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", 
           "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", 
           "she's": "she is", "should've": "should have", "shouldn't": "should not", 
           "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is",
           "that'd": "that would", "that'd've": "that would have", "that's": "that is", 
           "there'd": "there would", "there'd've": "there would have", "there's": "there is", 
           "here's": "here is","they'd": "they would", "they'd've": "they would have", 
           "they'll": "they will", "they'll've": "they will have", "they're": "they are", 
           "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", 
           "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", 
           "we're": "we are", "we've": "we have", "weren't": "were not", 
           "what'll": "what will", "what'll've": "what will have","what're": "what are",  
           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", 
           "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", 
           "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", 
           "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", 
           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", 
           "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have",
           "y'all're": "you all are","y'all've": "you all have","you'd": "you would", 
           "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", 
           "you're": "you are", "you've": "you have" }

def clean_text(text, lemmatize=True):
    wl = WordNetLemmatizer()
    stop = stopwords.words('english')
    soup = BeautifulSoup(text, "html.parser")  # Remove HTML tags
    text = soup.get_text()
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])  # Expand chatwords and contractions, clear contractions
    emoji_clean = re.compile("["
                             u"\U0001F600-\U0001F64F"  # Emoticons
                             u"\U0001F300-\U0001F5FF"  # Symbols & Pictographs
                             u"\U0001F680-\U0001F6FF"  # Transport & Map Symbols
                             u"\U0001F1E0-\U0001F1FF"  # Flags (iOS)
                             u"\U00002702-\U000027B0"
                             u"\U000024C2-\U0001F251"
                             "]+", flags=re.UNICODE)
    text = emoji_clean.sub(r'', text)
    text = re.sub(r'\.(?=\S)', '. ', text)  # Add space after full stop
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = "".join([word.lower() for word in text if word not in string.punctuation])  # Remove punctuation

    if lemmatize:
        text = " ".join([wl.lemmatize(word) for word in text.split() if word not in stop and word.isalpha()])  # Lemmatize
    else:
        text = " ".join([word for word in text.split() if word not in stop and word.isalpha()])
    return text

def data_processing(data):
    data['Dangerous permissions count'].fillna((data['Dangerous permissions count'].mean()), inplace=True)
    data['Description'].fillna('U', inplace=True)
    data['Related apps'].fillna('U', inplace=True)
    data.drop_duplicates(inplace=True)
    data.dropna(inplace=True)
    print(data.shape)
    data['Description'] = data['Description'].apply(clean_text, lemmatize=True)
    data['Package'] = data['Package'].apply(clean_text, lemmatize=True)
    return data

def train_data_visualization(data):
    print(data['Class'].value_counts())
    sns.countplot(data=data, x='Class')

    # Category with most benign
    cat_benign = data.groupby('Category')['Class'].apply(lambda x: (x == 0).sum()).reset_index(name='Benign').sort_values('Benign', ascending=False)
    plt.figure(figsize=(16, 8))
    sns.barplot(y="Category", x="Benign", data=cat_benign)

    # Category with most malware
    cat_malware = data.groupby('Category')['Class'].apply(lambda x: (x == 1).sum()).reset_index(name='Malware').sort_values('Malware', ascending=False)
    plt.figure(figsize=(16, 8))
    sns.barplot(y="Category", x="Malware", data=cat_malware)

    cat_all = pd.concat([cat_malware, cat_benign['Benign']], axis=1)
    t1 = go.Bar(
        y=cat_all['Malware'],
        x=cat_all['Category'],
        name='Malware',
        marker=dict(color='rgb(150,0,0)')
    )

    t2 = go.Bar(
        y=cat_all['Benign'],
        x=cat_all['Category'],
        name='Benign',
        marker=dict(color='rgb(0,150,0)')
    )

    d = [t1, t2]
    layout = go.Layout(
        title='Malware/Benign',
        barmode='stack',
        xaxis={'tickangle': -45},
        yaxis={'title': 'Malware/Benign'}
    )

    fig = go.Figure(data=d, layout=layout)

    ff.offline.iplot({'data': d, 'layout': layout})

    print(len(data['Package'].unique()))
    # Package with most benign
    print(data.groupby('Package')['Class'].apply(lambda x: (x == 0).sum()).reset_index(name='Benign').sort_values('Benign', ascending=False).head(20))

    # Package with most malware
    print(data.groupby('Package')['Class'].apply(lambda x: (x == 1).sum()).reset_index(name='Malware').sort_values('Malware', ascending=False).head(20))

    # Price and Class
    d = {'Price0_benign': [len(data[(data['Price'] == 0) & (data['Class'] == 0)])],
         'Price0_malware': [len(data[(data['Price'] == 0) & (data['Class'] == 1)])],
         'Price>0_benign': [len(data[(data['Price'] > 0) & (data['Class'] == 0)])],
         'Price>0_malware': [len(data[(data['Price'] > 0) & (data['Class'] == 1)])]}

    price_cls = pd.DataFrame(data=d)
    plt.figure(figsize=(16, 8))
    sns.barplot(data=price_cls)

    # Null values
    percent_null = data.isnull().mean().round(4) * 100
    trace = go.Bar(x=percent_null.index, y=percent_null.values, opacity=0.8, text=percent_null.values,
                   textposition='auto', marker=dict(color='#7EC0EE', line=dict(color='#000000', width=1.5)))

    layout = dict(title="Missing Values (count & %)")

    fig = dict(data=[trace], layout=layout)
    py.iplot(fig)

    # Word cloud for Description - malware
    malware_data = data[data.Class == 1]['Description']
    malware_data_string = ' '.join(malware_data)
    plt.figure(figsize=(20, 20))
    wc = WordCloud(max_words=2000, width=1200, height=600, background_color="white").generate(malware_data_string)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word cloud for Description - malware', fontsize=20)
    plt.show()

    # Word cloud for Description - benign
    benign_data = data[data.Class == 0]['Description']
    benign_data_string = ' '.join(benign_data)
    plt.figure(figsize=(20, 20))
    wc = WordCloud(max_words=2000, width=1200, height=600, background_color="white").generate(benign_data_string)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word cloud for Description - benign', fontsize=20)
    plt.show()

def ANN_model(X, X_train, y_train):
    # Initializing the ANN
    classifier = Sequential()
    classifier.add(Dense(units=len(X[0]), activation='relu'))
    classifier.add(Dense(units=40, activation='relu'))
    classifier.add(Dense(units=25, activation='relu'))
    classifier.add(Dense(units=12, activation='relu'))
    classifier.add(Dense(units=1, activation='sigmoid'))
    classifier.compile(Adam(lr=2e-5), loss='binary_crossentropy', metrics=['accuracy'])

    classifier.fit(X_train, y_train, batch_size=10, epochs=70)
    return classifier

def LR_model(X_test, y_test, X_train, y_train):
    regressor = LogisticRegression()
    regressor.fit(X_train, y_train)
    Y_pred = regressor.predict(X_test)
    pred_train = regressor.predict(X_train)
    print(accuracy_score(y_train, pred_train))
    print(accuracy_score(y_test, Y_pred))
    return regressor

def test_pre_processing(test_ann):
    test_ann = test_ann.drop("Package", axis=1)
    test_ann = test_ann.drop("Description", axis=1)
    test_ann = test_ann.drop("Related apps", axis=1)
    test_ann = test_ann.drop("App", axis=1)
    freq_port = test_ann['Dangerous permissions count'].dropna().mode()[0]
    test_ann['Dangerous permissions count'] = test_ann['Dangerous permissions count'].fillna(value=freq_port)
    encoder = LabelEncoder()
    test_ann['Category'] = encoder.fit_transform(test_ann['Category'])
    df_Category_one_hot2 = pd.get_dummies(test_ann['Category'], prefix='Category')
    result_test = pd.concat([df_Category_one_hot2, test_ann], axis=1)
    result_test.drop(['Category_29'], axis=1, inplace=True)
    result_test.drop(['Category'], axis=1, inplace=True)
    result_test.drop(['predicted_y'], axis=1, inplace=True)
    return result_test


df = load_data("C:\\Users\\revit\\OneDrive - Bar-Ilan University\\second year\\Semester A\\MACHINE LEARNING\\lior\\Android_train.csv")
data_info(df)
df = data_processing(df)
train_data_visualization(df)
encoder = LabelEncoder()
df['Category'] = encoder.fit_transform(df['Category'])
df_Category_one_hot = pd.get_dummies(df['Category'], prefix='Category')
result = pd.concat([df_Category_one_hot, df], axis=1)
result.drop(['Category_29'], axis=1, inplace=True)
result.drop(['Category'], axis=1, inplace=True)

s = result
s = s.drop("App", axis=1)
s = s.drop("Package", axis=1)
s = s.drop("Description", axis=1)
s = s.drop("Related apps", axis=1)
X = s.drop("Class", axis=1)
y = result["Class"]
sc = StandardScaler()
X = sc.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
classifier = ANN_model(X, X_train, y_train)
print(classifier.evaluate(X_test, y_test)[1])

test_ann = load_data("C:\\Users\\revit\\OneDrive - Bar-Ilan University\\second year\\Semester A\\MACHINE LEARNING\\lior\\Android_test.csv")
print(test_ann.isnull().sum())
test_ann = test_pre_processing(test_ann)
sc = StandardScaler()
X_ann_test = sc.fit_transform(test_ann)
ann_pred = classifier.predict(X_ann_test)
ann_pred = ann_pred.tolist()
ann_pred = [1 if i[0] > 0.5 else 0 for i in ann_pred]

regressor = LR_model(X_test, y_test, X_train, y_train)
sc = StandardScaler()
test_ann.drop(['predicted_y'], axis=1, inplace=True)
X_test4 = sc.fit_transform(test_ann)
LR_y = regressor.predict(X_test4)

test2 = load_data("C:\\Users\\revit\\OneDrive - Bar-Ilan University\\second year\\Semester A\\MACHINE LEARNING\\lior\\Android_test.csv")
test2.drop(['predicted_y'], axis=1, inplace=True)
test2['predicted_y'] = ann_pred
test2['predicted2_y'] = LR_y
test2.to_csv("C:\\Users\\revit\\OneDrive - Bar-Ilan University\\second year\\Semester A\\MACHINE LEARNING\\lior\\Android_test.csv", index=False)


