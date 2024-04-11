from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
from sklearn.preprocessing import StandardScaler  # כדי לבצע סטנדרטיזציה
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns  #ספרייה לייצור גרפיקה סטטיסטית
import pandas as pd

# ==========פונקציות ===========================

# לפתיחת הקובץ
def load_data(path):
    data = pd.read_csv(path)
    return data

# אינפורמציה על הנתונים 
def data_info(data):
    """
    Prints information about the dataset.
    """
    print(data.shape)
    print(data.head(5))
    print(data.info())
    print(data.describe())
    print(data.isnull().sum())

def data_visualization(data, save_path=None):
    """
    Visualizes the features of the dataset and saves the plot to a file if specified.
    """
    # Visualize the features of the dataframe
    data.hist(bins=50, figsize=(20, 15))
    plt.show()

    # Countplot of Deaths vs Survivals
    fig, ax = plt.subplots(1, figsize=(5, 5))
    plt.suptitle('Frequency of Survival', fontsize=15)
    ax.bar(data['Survived'].value_counts().index,
           data['Survived'].value_counts().values,
           color=['darkblue', 'darkorange'])
    ax.set_xticks(range(0, 2))
    ax.set_xticklabels(['Did not survive', 'Survived'], fontsize=14)
    plt.show()

    # Gender Split
    fig, ax = plt.subplots(1, figsize=(5, 5))
    plt.suptitle('Gender Split', fontsize=15)
    ax.bar(data['Sex'].value_counts().index,
           data['Sex'].value_counts().values,
           color=['darkblue', 'darkorange'])
    ax.set_xticks(range(0, 2))
    ax.set_xticklabels(['Male', 'Female'], fontsize=14)
    plt.show()

    # Countplot survival rate
    sns.catplot(x="Sex", hue="Survived", kind="count", data=data)
    plt.show()

    # Age Distribution of Passengers
    fig, ax = plt.subplots(figsize=(10, 7))
    age_died = data[data['Survived'] == 0]['Age']
    age_survive = data[data['Survived'] == 1]['Age']
    n, bins, patches = plt.hist(x=[age_died, age_survive],
                                stacked=True, bins='auto',
                                color=['darkblue', 'darkorange'],
                                alpha=0.65, rwidth=0.95)
    plt.grid(axis='y', alpha=0.55)
    plt.xlabel('Passenger Age', fontsize=15)
    plt.ylabel('Frequency', fontsize=15)
    plt.title('Age Distribution of Passengers', fontsize=22)
    plt.legend(['Died', 'Survived'], fontsize=15)
    plt.show()

    # Pclass vs Survived
    group = data.groupby(['Pclass', 'Survived'])
    pclass_survived = group.size().unstack()
    sns.heatmap(pclass_survived, annot=True, fmt="d")
    plt.show()

    # Categorical Count Plots for Embarked
    sns.catplot(x='Embarked', hue='Survived', kind='count', col='Pclass', data=data)
    plt.show()

    # Save the plot to a file if specified
    if save_path:
        fig.savefig(save_path)

def ticket_check(tic):
    """
    Cleans the ticket information.
    """
    new_tic = []
    counter = 0
    for ticet in tic:
        x = ticet.split(" ")
        x = x[0]
        if x[-1] == ".":
            x = x[:-1]
        if x.isdigit() or x is None:
            x = 'U'
            counter += 1
        new_tic.append(x)
    print(counter)
    return new_tic

def pre_processing(data):
    """
    Performs data preprocessing on the dataset.
    """
    mean = data["Age"].mean()
    data['Age'].fillna(mean, inplace=True)
    print(data.isnull().sum())
    data['Cabin'].fillna('U', inplace=True)
    data['Cabin'] = data['Cabin'].apply(lambda x: x[0])
    print(data.isnull().sum())
    print(data['Cabin'].value_counts())
    data['Sex'] = data['Sex'].map({'female': 1, 'male': 0}).astype(int)
    freq_port = data.Embarked.dropna().mode()[0]
    data['Embarked'] = data['Embarked'].fillna(value=freq_port)
    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    data.drop(['Name', 'PassengerId'], axis=1, inplace=True)
    tic = data['Ticket'].to_list()
    tic = ticket_check(tic)
    data.drop(['Ticket'], axis=1, inplace=True)
    data['Cabin'].unique()
    data_values = data.iloc[:, :].values
    labelencoder = LabelEncoder()
    data_values[:, -1] = labelencoder.fit_transform(data_values[:, -1])
    data_values[:, 7] = labelencoder.fit_transform(data_values[:, 7])
    df_plcass_one_hot = pd.get_dummies(data['Pclass'], prefix='pclass')
    df_Cabin_one_hot = pd.get_dummies(data['Cabin'], prefix='Cabin')
    result = pd.concat([data, df_plcass_one_hot], axis=1)
    result = pd.concat([result, df_Cabin_one_hot], axis=1)
    result.drop('Pclass', inplace=True, axis=1)
    result.drop('Cabin', inplace=True, axis=1)
    result.drop('pclass_3', inplace=True, axis=1)
    result.drop('Cabin_U', inplace=True, axis=1)
   # sc = StandardScaler()F
   # X = sc.fit_transform(result)
    return result

def train_lr_model(data):
    """
    Trains a Logistic Regression model on the data.
    """
    X = data.drop("Survived", axis=1)
    y = data["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    regressor = LogisticRegression()
    regressor.fit(X_train, y_train)
    Y_pred = regressor.predict(X_test)
    pred_train = regressor.predict(X_train)
    print("Training Accuracy:", accuracy_score(y_train, pred_train))
    print("Testing Accuracy:", accuracy_score(y_test, Y_pred))
    return regressor

def train_rf_model(data):
    """
    Trains a Random Forest Classifier model on the data.
    """
    X = data.drop("Survived", axis=1)
    y = data["Survived"]
    X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    rf_clf = RandomForestClassifier(n_estimators=90, max_depth=10, min_samples_leaf=2, random_state=42)
    rf_clf.fit(X_train, y_train)
    pred_train = rf_clf.predict(X_train)
    pred_test = rf_clf.predict(X_test)
    train_accuracy = accuracy_score(y_train, pred_train)
    test_accuracy = accuracy_score(y_test, pred_test)
    print("Training Accuracy:", train_accuracy)
    print("Testing Accuracy:", test_accuracy)
    return rf_clf

df = load_data("C:\\Users\\revit\\OneDrive - Bar-Ilan University\\second year\\Semester A\\MACHINE LEARNING\\lior\\Titanic_train.csv")

data_info(df) #הוספתי כדי להדפיס את הנתונים 

women = df.loc[df.Sex == 'female']["Survived"]
rate_women = sum(women) / len(women)
men = df.loc[df.Sex == 'male']["Survived"]
rate_men = sum(men) / len(men)
print("% of men survivors:", rate_men)
print("% of women survivors:", rate_women)

data_visualization(df, save_path='visualization.png')

train = pre_processing(df)
lr_regressor = train_lr_model(train)
rf_clf = train_rf_model(train)

df = load_data("C:\\Users\\revit\\OneDrive - Bar-Ilan University\\second year\\Semester A\\MACHINE LEARNING\\lior\\Titanic_test.csv")
test_df = df
data_visualization(df, save_path='visualization_test.png')

test_df = pre_processing(test_df)

test_df.drop('LR_y', inplace=True, axis=1)
test_df.drop('RF_y', inplace=True, axis=1)

sc = StandardScaler()
X_1 = sc.fit_transform(test_df)
LR_y = lr_regressor.predict(X_1)
RF_y = rf_clf.predict(X_1)
df['LR_y'] = LR_y
df['RF_y'] = RF_y
df.to_csv("Titanic_test.csv", index=False)