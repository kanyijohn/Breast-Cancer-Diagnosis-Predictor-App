import pandas as pd #pip install pandas
from sklearn.preprocessing import StandardScaler #standardize features
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report #accuracy score and detailed metrics report
import pickle as pickle #used for serializing and deserializing Python objects into a format that can be easily stored or transmitted and deserialization is the reverse process

def create_model(data): # creating the model

    X = data.drop(['diagnosis'], axis=1) #predictors (independent variables)-all columns except diagnosis 
    y = data['diagnosis'] #target variable
    
    # scaling the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X) #standardize the X input features

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
  )
    
    # train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # test model
    y_pred = model.predict(X_test)
    print('Accuracy of our model: ', accuracy_score(y_test, y_pred))
    print("Classification report: \n", classification_report(y_test, y_pred))
  
    return model, scaler




def get_clean_data(): # loading and cleaning the data

    data = pd.read_csv("data/data.csv") #reading the dataset
    
    data = data.drop(['Unnamed: 32', 'id'], axis=1) # removing the Unnamed column from row 1

    data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 }) # transforming the diagnosis to 1 for Malicicous(M) and 0 for Benign (B)
    
    return data




# main function( running the main function- the functions contained in the main function are being called for execution under the main function)
def main():
    data = get_clean_data() 
    
    model, scaler = create_model(data)
    
    with open('model/model.pkl', 'wb') as f:
     pickle.dump(model, f) # dumps the model object (i.e logical regression model) and writes it to the file f. The model is then saved as a binary .pkl file.
    
    with open('model/scaler.pkl', 'wb') as f:
     pickle.dump(scaler, f) # dumps the scaler object and writes it to the file f. The model is then saved as a binary .pkl file.
    






if __name__ == '__main__': #helps code structure such that certain parts (like main()) only run when the script is executed directly, not when it is imported
  main()

