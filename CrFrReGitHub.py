## https://www.kaggle.com/c/GiveMeSomeCredit/data
## https://www.youtube.com/watch?v=yX8KuPZCAMo

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from scipy import stats

##### Creating charts
def graphs_charts(df):
    print(pd.crosstab(df['SeriousDlqin2yrs'], columns="count"))
    plt.show(df.boxplot('age', by='SeriousDlqin2yrs'))
    plt.show(df.boxplot('NumberOfTime30-59DaysPastDueNotWorse', by='SeriousDlqin2yrs'))
    plt.show(df.boxplot('DebtRatio', by='SeriousDlqin2yrs'))
    plt.show(df.boxplot('MonthlyIncome', by='SeriousDlqin2yrs'))
    plt.show(df.boxplot('NumberOfOpenCreditLinesAndLoans', by='SeriousDlqin2yrs'))
    plt.show(df.boxplot('NumberOfTimes90DaysLate', by='SeriousDlqin2yrs'))
    plt.show(df.boxplot('NumberRealEstateLoansOrLines', by='SeriousDlqin2yrs'))
    plt.show(df.boxplot('NumberOfTime60-89DaysPastDueNotWorse', by='SeriousDlqin2yrs'))
    plt.show(df.boxplot('NumberOfDependents', by='SeriousDlqin2yrs'))


##### Reading the data set
def read_dataset():
    df = pd.read_csv("./cs_training.csv")
    print(df.describe()) ## Describe the data set
    df=df.fillna(0) ## Assign All NaN to zero
    print('Import dataframe shape',df.shape)  ##datframe size
    print(df.head(2))  ##first 5 rows of the data frmae
    print(list(df))  ##column names
    ########## Adjust sammple to balance the data
    df_majority = df[df.SeriousDlqin2yrs == 'N'] ##Identifying majority
    df_minority = df[df.SeriousDlqin2yrs == 'Y'] ##Identifying minority
    print('majority', df_majority.shape)
    print('majority', df_minority.shape)
    print('********************')
    # Downward the majority class
    df_majority_resample = resample(df_majority, replace=False, n_samples=16000, random_state=123)
    print('new maj sample', df_majority_resample.shape)
    df_downsample = pd.concat([df_majority_resample, df_minority])
    print('Down sample final size', df_downsample.shape)
    print('************')
    #graphs_charts(df_downsample)
    df_Final = df_downsample[(np.abs(stats.zscore(df_downsample[df_downsample.columns[1:11]])) < 2).all(axis=1)]
    print('After Removing Outliers', df_Final.shape)
    #graphs_charts(df_Final)
   #########33
    x = df_Final[df_Final.columns[1:11]] ##columns = ['b', 'c']  ,df1 = pd.DataFrame(df, columns=columns) getting columns
    y = df_Final[df_Final.columns[0]] ##columns = ['b', 'c']  ,df1 = pd.DataFrame(df, columns=columns) getting columns
    encoder=LabelEncoder()
    encoder.fit(y)
    y=encoder.transform(y)
    Y=one_hot_encode(y)
    print(x.shape)
    return(x,Y)

# define the Encorder function
def one_hot_encode(labels):
    n_labels=len(labels)
    n_unique_labels=len(np.unique(labels))
    one_hot_encode=np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels),labels]=1
    return one_hot_encode

#  read the data set
x,Y=read_dataset()

#shuffle the dataset to mix up the rows
x,Y=shuffle(x,Y,random_state=10)

# convert data set into training and testing data set
train_x,test_x,train_y,test_y=train_test_split(x,Y,test_size=0.20,random_state=42)


# Scaled to a small range like 0 to 1
scaler = MinMaxScaler(feature_range=(0, 1))

# Scale both the training & testing data
train_x1 = scaler.fit_transform(train_x.astype(np.float32))
test_x1 = scaler.fit_transform(test_x.astype(np.float32))

# Inspect the shape of the training and testing
print(train_x.shape)
print(train_x1.shape)
print(train_y.shape)
print(test_x.shape)
print(test_x1.shape)
print(test_y.shape)

# Define parameters and variables
learning_rate=0.05
training_epochs=1000
cost_history=np.empty(shape=[1],dtype=float)
n_dim=x.shape[1]
print("n_dim",n_dim)
n_class=2
model_path="D:/Sajith/test1/CrForcat/test"

#Define number of hidden layers and neurons for each layer
n_hidden_1=10
n_hidden_2=50
n_hidden_3=60
n_hidden_4=20

x = tf.placeholder(tf.float32,[None,n_dim])
W = tf.Variable(tf.zeros([n_dim,n_class]))
b = tf.Variable(tf.zeros([n_class]))
y_=tf.placeholder(tf.float32,[None,n_class])

# Define the model
def multilayer_perception(x,weights,biases):
    layer_1=tf.add(tf.matmul(x,weights['h1']),biases['b1'])
    layer_1=tf.nn.tanh(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.tanh(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.tanh(layer_3)

    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.tanh(layer_4)

    #output layer with linear activation
    out_layer=tf.matmul(layer_4,weights['out'])+biases['out']
    return out_layer

#Weights
weights={
    'h1':tf.Variable(tf.truncated_normal([n_dim,n_hidden_1])),
    'h2':tf.Variable(tf.truncated_normal([n_hidden_1,n_hidden_2])),
    'h3':tf.Variable(tf.truncated_normal([n_hidden_2,n_hidden_3])),
    'h4':tf.Variable(tf.truncated_normal([n_hidden_3,n_hidden_4])),
    'out':tf.Variable(tf.truncated_normal([n_hidden_4,n_class]))
    }
#Biases
biases={
        'b1':tf.Variable(tf.truncated_normal([n_hidden_1])),
        'b2':tf.Variable(tf.truncated_normal([n_hidden_2])),
        'b3':tf.Variable(tf.truncated_normal([n_hidden_3])),
        'b4':tf.Variable(tf.truncated_normal([n_hidden_4])),
        'out':tf.Variable(tf.truncated_normal([n_class]))
        }

# Initialize all the variables
init=tf.global_variables_initializer()
saver=tf.train.Saver()

# Call the model defined
y=multilayer_perception(x,weights,biases)

#Cost function
cost_function=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y,labels=y_))
#Optimizer
training_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

sess=tf.Session()
sess.run(init)

#calculate the cost and the accuracy for each epoch
mse_history=[]
accuracy_history=[]

for epoch in range(training_epochs):
    sess.run(training_step,feed_dict={x:train_x1,y_:train_y})
    cost=sess.run(cost_function,feed_dict={x:train_x1,y_:train_y})
    cost_history=np.append(cost_history,cost)
    correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    pred_y=sess.run(y,feed_dict={x:test_x1})
    mse=tf.reduce_mean(tf.square(pred_y-test_y))
    mse_=sess.run(mse)
    mse_history.append(mse_)
    accuracy=(sess.run(accuracy,feed_dict={x:train_x1,y_:train_y}))
    accuracy_history.append(accuracy)

    print('epoch:',epoch,'-','cost',cost,"-MSE",mse,"-Train Accuracy",accuracy)

save_path=saver.save(sess,model_path)
print("Model saved in the file %s" % save_path)

#MSE and accuracy grpah
plt.show(plt.plot(mse_history,'r'))
plt.show(plt.plot(accuracy_history))

#print the final accuracy
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print('Test accuracy: ',sess.run(accuracy,feed_dict={x:test_x1,y_:test_y}))
print('***********')

#print final mse
pred_y=sess.run(y,feed_dict={x:test_x1})
mse=tf.reduce_mean(tf.square(pred_y-test_y))
print("MSE : %.4f" % sess.run(mse))

