import pickle
from sklearn.externals import joblib
import codecs
from collections import defaultdict
import csv
import sys
import csv
import codecs
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score
import pickle
from sklearn.externals import joblib
from sklearn.datasets import load_files
import math
from sklearn import datasets, linear_model
from sklearn import svm
from sklearn.metrics import accuracy_score
csv.field_size_limit(500000)
director_fb_likes = []
num_critic_for_reviews = []
actor3_fb_likes = []
actor1_fb_likes = []
gross = []
fb_likes = []
duration = []
num_voted_users = []
num_user_for_reviews = []
language  = []
country = []
content_rating  = []
budget  = []
year = []
actor2_fb_likes  = []
imdb_score = []
total_facebook_likes = []
#Movie Genres
genres = []
action = [0 for x in range(5043)]
adventure = [0 for x in range(5043)]
fantasy = [0 for x in range(5043)]
scifi = [0 for x in range(5043)]
thriller = [0 for x in range(5043)]
comedy = [0 for x in range(5043)]
family = [0 for x in range(5043)]
horror = [0 for x in range(5043)]
war = [0 for x in range(5043)]
animation = [0 for x in range(5043)]
western = [0 for x in range(5043)]
romance = [0 for x in range(5043)]
musical = [0 for x in range(5043)]
documentary = [0 for x in range(5043)]
drama = [0 for x in range(5043)]
history = [0 for x in range(5043)]
biography = [0 for x in range(5043)]
mystery = [0 for x in range(5043)]
crime = [0 for x in range(5043)]
#content ratings
g = [0 for x in range(5043)]
pg = [0 for x in range(5043)]
pg_13 = [0 for x in range(5043)]
r = [0 for x in range(5043)]
nc_17 = [0 for x in range(5043)]
with open("movie_metadata.csv", "r") as sentences_file:
    reader = csv.reader(sentences_file, delimiter=',')
    reader.next()
    i = 0
    for row in reader:
        content_rating = ""
        content_rating = str(row[21])
        if 'G' in content_rating:
            g[i] = 1
        if 'PG' in content_rating:
            pg[i] = 1
        if 'PG-13' in content_rating:
            pg_13[i] = 1
        if 'R' in content_rating:
            r[i] = 1
        if 'NC-17' in content_rating:
            nc_17[i] = 1
        i+=1
fil = open("writeFile.txt","w")
for lan in nc_17:
    fil.write("%s"%(str(lan)))
    fil.write('\n')
fil.close()
with open("movie_metadata.csv", "r") as sentences_file:
    reader = csv.reader(sentences_file, delimiter=',')
    reader.next()
    i = 0
    for row in reader:
        genre = ""
        genre = str(row[9])
        if 'Action' in genre:
            action[i] = 1
        if 'Adventure' in genre:
            adventure[i] = 1
        if 'Fantasy' in genre:
            fantasy[i] = 1
        if 'Sci-Fi' in genre:
            scifi[i] = 1
        if 'Thriller' in genre:
            thriller[i] = 1
        if 'Comedy' in genre:
            comedy[i] = 1
        if 'Family' in genre:
            family[i] = 1
        if 'Horror' in genre:
            horror[i] = 1
        if 'War' in genre:
            war[i] = 1
        if 'Animation' in genre:
            animation[i] = 1
        if 'Western' in genre:
            western[i] = 1
        if 'Romance' in genre:
            romance[i] = 1
        if 'Musical' in genre:
            musical[i] = 1
        if 'Documentary' in genre:
            documentary[i] = 1
        if 'Drama' in genre:
            drama[i] = 1
        if 'History' in genre:
            history[i] = 1
        if 'Biography' in genre:
            biography[i] = 1
        if 'Mystery' in genre:
            mystery[i] = 1
        if 'Crime' in genre:
            crime[i] = 1
        i+=1
        
# fil = open("writeFile.txt","w")
# for lan in crime:
#     fil.write("%s"%(str(lan)))
#     fil.write('\n')
# fil.close()
with open("movie_metadata.csv", "r") as sentences_file:
    reader = csv.reader(sentences_file, delimiter=',')
    reader.next()
    i = 1
    for row in reader:
        director_fb_likes.append(float(row[4]))
        num_critic_for_reviews.append(float(row[2]))
        actor1_fb_likes.append(float(row[7]))
        actor3_fb_likes.append(float(row[5]))
        fb_likes.append(float(row[13]))
        duration.append(float(row[3]))
        if int(row[8])>0:
            gross.append(int(math.log(float(row[8]))/math.log(10.0)))
        else:
            gross.append(0)
        num_voted_users.append(float(row[12]))
        num_user_for_reviews.append(float(row[18]))
        if row[19] == 'English':
            language.append(1.0)
        else:
            language.append(0.0)
        if row[20] == 'USA':
            country.append(1.0)
        else:
            country.append(0.0)
        budget.append(float(row[22]))
        actor2_fb_likes.append(float(row[24]))
        imdb_score.append(float(row[25]))
        total_facebook_likes.append(float(row[27]))
# print(len(total_facebook_likes))
for i in range(0,5043):
    director_fb_likes[i]/=max(director_fb_likes)
    num_critic_for_reviews[i]/=max(num_critic_for_reviews)
    actor1_fb_likes[i]/=max(actor1_fb_likes)
    actor2_fb_likes[i]/=max(actor2_fb_likes)
    actor3_fb_likes[i]/=max(actor3_fb_likes)
    fb_likes[i]/=max(fb_likes)
    duration[i]/=max(duration)
    num_voted_users[i]/=max(num_voted_users)
    num_user_for_reviews[i]/=max(num_user_for_reviews)
    budget[i]/=max(budget)
    imdb_score[i]/=max(imdb_score)
    total_facebook_likes[i]/=max(total_facebook_likes)
num_row = 5043
num_col = 35
totalMatrix = [[0 for x in range(num_col)] for y in range(num_row)]
for i in range(0,5043):
    for j in range(0,35):
        if j==0:
            totalMatrix[i][j] = director_fb_likes[i]
        elif j==1:
            totalMatrix[i][j] = num_critic_for_reviews[i]
        elif j==2:
            totalMatrix[i][j] = actor1_fb_likes[i]
        elif j==3:
            totalMatrix[i][j] = actor2_fb_likes[i]
        elif j==4:
            totalMatrix[i][j] = actor3_fb_likes[i]
        elif j==5:
            totalMatrix[i][j] = fb_likes[i]
        elif j==6:
            totalMatrix[i][j] = duration[i]
        elif j==7:
            totalMatrix[i][j] = num_voted_users[i]
        elif j==8:
            totalMatrix[i][j] = num_user_for_reviews[i]
        elif j==9:
            totalMatrix[i][j] = budget[i]
        elif j==10:
            totalMatrix[i][j] = imdb_score[i]
        elif j==11:
            totalMatrix[i][j] = total_facebook_likes[i]
        elif j==12:
            totalMatrix[i][j] = action[i]
        elif j==13:
            totalMatrix[i][j] = adventure[i]
        elif j==14:
            totalMatrix[i][j] = fantasy[i]
        elif j==15:
            totalMatrix[i][j] = scifi[i]
        elif j==16:
            totalMatrix[i][j] = thriller[i]
        elif j==17:
            totalMatrix[i][j] = comedy[i]
        elif j==18:
            totalMatrix[i][j] = family[i]
        elif j==19:
            totalMatrix[i][j] = horror[i]
        elif j==20:
            totalMatrix[i][j] = war[i]
        elif j==21:
            totalMatrix[i][j] = animation[i]
        elif j==22:
            totalMatrix[i][j] = western[i]
        elif j==23:
            totalMatrix[i][j] = romance[i]
        elif j==24:
            totalMatrix[i][j] = musical[i]
        elif j==25:
            totalMatrix[i][j] = documentary[i]
        elif j==26:
            totalMatrix[i][j] = drama[i]
        elif j==27:
            totalMatrix[i][j] = history[i]
        elif j==28:
            totalMatrix[i][j] = biography[i]
        elif j==29:
            totalMatrix[i][j] = mystery[i]
        elif j==30:
            totalMatrix[i][j] = crime[i]
        elif j==31:
            totalMatrix[i][j] = g[i]
        elif j==32:
            totalMatrix[i][j] = pg[i]
        elif j==33:
            totalMatrix[i][j] = pg_13[i]
        elif j==34:
            totalMatrix[i][j] = r[i]
        elif j==35:
            totalMatrix[i][j] = nc_17[i]
trainMatrix = totalMatrix[:4539]
testMatrix = totalMatrix[4539:]
trainLabel = gross[:4539]
testLabel = gross[4539:]
svm_model = svm.SVC(kernel='rbf', C=2, gamma = 2)
svm_model2 = svm.SVC(kernel='linear', C=2, gamma = 2)
svm_model3 = svm.SVC(kernel='poly', C=1, gamma = 2)
from sklearn import metrics
from sklearn.model_selection import cross_val_score
scores2 = cross_val_score(svm_model, totalMatrix, gross, cv=5)
print("SVM_rbf")
print(scores2)
scores3 = cross_val_score(svm_model2, totalMatrix, gross, cv=5)
print("SVM linear")
print(scores3)
scores4 = cross_val_score(svm_model3, totalMatrix, gross, cv=5)
print("SVM poly")
print(scores4)
from sklearn.linear_model import LogisticRegression
logreg_model = LogisticRegression(C=1e5)
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
scores5 = cross_val_score(clf, totalMatrix, gross, cv=5)
print("NB")
print(scores5)
from sklearn.neural_network import MLPClassifier
mlp_model = MLPClassifier(hidden_layer_sizes=(15, ), activation='identity',  max_iter=200)
scores6 = cross_val_score(mlp_model, totalMatrix, gross, cv=5)
print("mlp1")
print(scores6)
mlp_model2 = MLPClassifier(hidden_layer_sizes=(15, ), activation='logistic',  max_iter=200)
scores7 = cross_val_score(mlp_model2, totalMatrix, gross, cv=5)
print("mlp2")
print(scores7)
mlp_model3 = MLPClassifier(hidden_layer_sizes=(15, ), activation='tanh',  max_iter=200)
scores8 = cross_val_score(mlp_model3, totalMatrix, gross, cv=5)
print("mlp3")
print(scores8)
mlp_model4 = MLPClassifier(hidden_layer_sizes=(15, ), activation='relu',  max_iter=200)
scores9 = cross_val_score(mlp_model4, totalMatrix, gross, cv=5)
print("mlp4")
print(scores9)
mlp_model5 = MLPClassifier(hidden_layer_sizes=(20, ), activation='identity',  max_iter=200)
scores10 = cross_val_score(mlp_model5, totalMatrix, gross, cv=5)
print("mlp5")
print(scores10)
mlp_model6 = MLPClassifier(hidden_layer_sizes=(20, ), activation='logistic',  max_iter=200)
scores11 = cross_val_score(mlp_model6, totalMatrix, gross, cv=5)
print("mlp6")
print(scores11)
mlp_model7 = MLPClassifier(hidden_layer_sizes=(20, ), activation='tanh',  max_iter=200)
scores12 = cross_val_score(mlp_model7, totalMatrix, gross, cv=5)
print("mlp7")
print(scores12)
mlp_model8 = MLPClassifier(hidden_layer_sizes=(20, ), activation='relu',  max_iter=200)
scores13 = cross_val_score(mlp_model8, totalMatrix, gross, cv=5)
print("mlp8")
print(scores13)
logreg_model.fit(trainMatrix,trainLabel)
predicted = logreg_model.predict(testMatrix)
count = 0
for i in range(len(predicted)):
    if (predicted[i]-testLabel[i])==0:
        count += 1
print("logistic Regression")
print(float(count)/float(len(predicted)))
scores = cross_val_score(logreg_model, totalMatrix, gross, cv=5)
print(scores)
print("now")
print("max_gross")
print(max(gross))
print("mingross")
print(min(gross))
from sklearn.cross_validation import train_test_split
for i in range(5):
    x_train, x_test, y_train, y_test = train_test_split(totalMatrix, gross, test_size=0.3)
    # logreg_model2 = LogisticRegression(C=1e5)
    # logreg_model2.fit(x_train,y_train)
    # predicted2 = logreg_model2.predict(x_test)
    mlp_model8.fit(x_train,y_train)
    predicted2 = mlp_model8.predict(x_test)
    count2 = 0
    for i in range(len(predicted2)):
        if (predicted2[i]-y_test[i])==0 or abs(predicted2[i]-y_test[i])==1:
            count2 += 1
    print(float(count2)/float(len(predicted2)))
from sklearn import linear_model
lr = linear_model.LinearRegression()
predictedLinear = cross_val_score(lr, totalMatrix, gross, cv=5)
print("linear regression")
print(predictedLinear)
