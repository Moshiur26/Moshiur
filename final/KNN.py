from sklearn import datasets
import numpy as np
import math
from sklearn.model_selection import train_test_split


iris = datasets.load_iris()
X = iris.data[:, :2] 
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33,random_state=42)

train_set_size = len(y_train)
test_set_size = len(y_test)

distances = np.zeros((1,train_set_size))
number_of_neighbors = 3
number_of_classes = len(np.unique(y_train))
predictions = np.zeros((test_set_size,1))


for i in range(test_set_size) :
    x_1 = X_test[i,:]
    for j in range(train_set_size) :
        x_2 = X_train[j,:]
        #get euclidean distance between x_1 and x_2
        distances[0,j] = math.sqrt(np.sum((x_1 - x_2)**2))
    #find nearest neighbors
    neighbors = distances[0,:].argsort()[:number_of_neighbors]
    #find out the majorioty class
    
    markers = np.zeros((1,number_of_classes))
    
    for j in range(number_of_neighbors) :
        markers[0,y_train[neighbors[j]]] = markers[0,y_train[neighbors[j]]] + 1 
        
    #find out the class with max count
    predicted_class = np.argmax(markers[0,:])
    
    #perform classification here
    predictions[i] = predicted_class
    
    
    
    
#find out the accuracy
number_of_test_instances_correctly_classified = np.count_nonzero(predictions[:,0]==y_test)
print("So the accuracy of my KNN classifier is : ",(number_of_test_instances_correctly_classified/test_set_size)*100,"percent")        
    



        

        