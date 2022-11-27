import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
 
# Reading the data
TicketData=pd.read_csv("D:\\Emotion analysis\\testing\\plutchik.csv", encoding='latin-1')
 
# Printing number of rows and columns
print(TicketData.shape)
 
# Printing sample rows
TicketData.head(10)

# Number of unique values for urgency column
# You can see there are 3 ticket types
print(TicketData.groupby('emotions').size())

# Plotting the bar chart
%matplotlib inline
TicketData.groupby('emotions').size().plot(kind='bar');

# Count vectorization of text
from sklearn.feature_extraction.text import CountVectorizer
 
# Ticket Data
corpus = TicketData['text'].values
 
# Creating the vectorizer
vectorizer = CountVectorizer(stop_words='english')
 
# Converting the text to numeric data
X = vectorizer.fit_transform(corpus)
 
#print(vectorizer.get_feature_names())
 
# Preparing Data frame For machine learning
# Priority column acts as a target variable and other columns as predictors
CountVectorizedData=pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
CountVectorizedData['Priority']=TicketData['emotions']
print(CountVectorizedData.shape)
CountVectorizedData.head()


# Defining an empty dictionary to store the values
GloveWordVectors = {}
 
# Reading Glove Data
with open("D:\\Emotion analysis\\testing\\glove.6B.100d.txt", 'r', encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.array(values[1:], "float")
        GloveWordVectors[word] = vector
        
# Checking the total number of words present
len(GloveWordVectors.keys())

# Each word has a numeric representation of 100 numbers
GloveWordVectors['bye'].shape

# Sample glove word vector
GloveWordVectors['bye']

# Creating the list of words which are present in the Document term matrix
WordsVocab=CountVectorizedData.columns[:-1]
 # Printing sample words
WordsVocab[0:10]

def FunctionText2Vec(inpTextData):
    # Converting the text to numeric data
    X = vectorizer.transform(inpTextData)
    CountVecData=pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
    
    # Creating empty dataframe to hold sentences
    W2Vec_Data=pd.DataFrame()
    
    # Looping through each row for the data
    for i in range(CountVecData.shape[0]):
 
        # initiating a sentence with all zeros
        Sentence = np.zeros(100)
 
        # Looping thru each word in the sentence and if its present in 
        # the Glove model then storing its vector
        for word in WordsVocab[CountVecData.iloc[i,:]>=1]:
            #print(word)
            if word in GloveWordVectors.keys():    
                Sentence=Sentence+GloveWordVectors[word]
        # Appending the sentence to the dataframe
        W2Vec_Data=W2Vec_Data.append(pd.DataFrame([Sentence]))
    return(W2Vec_Data)
    
   # Since there are so many words... This will take some time :( 
# Calling the function to convert all the text data to Glove Vectors
W2Vec_Data=FunctionText2Vec(TicketData['text'])
 
# Checking the new representation for sentences
W2Vec_Data.shape

# Comparing the above with the document term matrix
CountVectorizedData.shape

# Adding the target variable
W2Vec_Data.reset_index(inplace=True, drop=True)
W2Vec_Data['Priority']=CountVectorizedData['Priority']
 
# Assigning to DataForML variable
DataForML=W2Vec_Data
DataForML.head()

# Separate Target Variable and Predictor Variables
TargetVariable=DataForML.columns[-1]
Predictors=DataForML.columns[:-1]
 
X=DataForML[Predictors].values
y=DataForML[TargetVariable].values
 
# Split the data into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=428)
 
# Sanity check for the sampled data
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

from sklearn.preprocessing import StandardScaler, MinMaxScaler
# Choose either standardization or Normalization
# On this data Min Max Normalization is used because we need to fit Naive Bayes
 
# Choose between standardization and MinMAx normalization
#PredictorScaler=StandardScaler()
PredictorScaler=MinMaxScaler()
 
# Storing the fit object for later reference
PredictorScalerFit=PredictorScaler.fit(X)
 
# Generating the standardized values of X
X=PredictorScalerFit.transform(X)
 
# Split the data into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=428)
 
# Sanity check for the sampled data
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# K-Nearest Neighbor(KNN)
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=15)
 
# Printing all the parameters of KNN
print(clf)
 
# Creating the model on Training Data
KNN=clf.fit(X_train,y_train)
prediction=KNN.predict(X_test)
 
# Measuring accuracy on Testing Data
from sklearn import metrics
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))
 
# Printing the Overall Accuracy of the model
F1_Score=metrics.f1_score(y_test, prediction, average='weighted')
print('Accuracy of the model on Testing Sample Data:', round(F1_Score,2))
 
# Importing cross validation function from sklearn
from sklearn.model_selection import cross_val_score
 
# Running 10-Fold Cross validation on a given algorithm
# Passing full data X and y because the K-fold will split the data and automatically choose train/test
#Accuracy_Values=cross_val_score(KNN, X , y, cv=10, scoring='f1_weighted')
#print('\nAccuracy values for 10-fold Cross Validation:\n',Accuracy_Values)
#print('\nFinal Average Accuracy of the model:', round(Accuracy_Values.mean(),2))
 
# Plotting the feature importance for Top 10 most important columns
# There is no built-in method to get feature importance in KNN
