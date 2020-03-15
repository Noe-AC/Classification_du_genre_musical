# On importe les données dans un array numpy
import pandas as pd
import numpy as np
#data_path = "4ieme_sec_non_ajuste"
data_path = "12ieme_sec_ajuste"
Xy = (pd.read_csv(data_path + "/data_cla_ele_met.csv",sep = ',',skipinitialspace=False)).to_numpy()
# On découpe les données en features et target
id_data = Xy[:,0] 
y       = Xy[:,1]  # le target : genre = {0,1,2}
X       = Xy[:,2:] # features : spectre

# On découpe les données en train + test
from sklearn.model_selection import train_test_split
test_size=0.25
random_state = 27
Xy_train, Xy_test = train_test_split(Xy,test_size=test_size, random_state=random_state)
id_train = Xy_train[:,0]
y_train  = Xy_train[:,1]
X_train  = Xy_train[:,2:]
id_test  = Xy_test[:,0]
y_test   = Xy_test[:,1]
X_test   = Xy_test[:,2:]
#print(X.shape,X_train.shape,X_test.shape,y.shape,y_train.shape,y_test.shape) # donne : (1068, 464) (801, 464) (267, 464) (1068,) (801,) (267,)
print("\n")
print("X.shape :\t",X.shape)
print("X_train.shape :\t",X_train.shape)
print("X_test.shape :\t",X_test.shape)
print("y.shape :\t",y.shape)
print("y_train.shape :\t",y_train.shape)
print("y_test.shape :\t",y_test.shape)
print("\n")

# Paramètres d'entraînement
n_neighbors = 1
kernel = 'rbf'
SVC_gamma = 0.001
#SVC_C = 4.9
SVC_C = 100.

# On met les erreurs à OFF
import warnings
warnings.simplefilter("ignore")


#####################################
############### KNN #################

# Apprentissage : KNN
from sklearn.neighbors import KNeighborsClassifier

n_neighbors = 1
clf = KNeighborsClassifier(n_neighbors=n_neighbors)
clf.fit(X_train,y_train)
print("KNN :\t train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train,y_train)*100 , clf.score(X_test,y_test)*100 ))
#print("KNN (k=%d)"%n_neighbors,"\t : train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train,y_train)*100 , clf.score(X_test,y_test)*100 ))
# Le meilleur est à k=1

"""
n_neighbors = 2
clf = KNeighborsClassifier(n_neighbors=n_neighbors)
clf.fit(X_train,y_train)
print("KNN (k=%d)"%n_neighbors,"\t : train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train,y_train)*100 , clf.score(X_test,y_test)*100 ))

n_neighbors = 3
clf = KNeighborsClassifier(n_neighbors=n_neighbors)
clf.fit(X_train,y_train)
print("KNN (k=%d)"%n_neighbors,"\t : train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train,y_train)*100 , clf.score(X_test,y_test)*100 ))

n_neighbors = 4
clf = KNeighborsClassifier(n_neighbors=n_neighbors)
clf.fit(X_train,y_train)
print("KNN (k=%d)"%n_neighbors,"\t : train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train,y_train)*100 , clf.score(X_test,y_test)*100 ))

n_neighbors = 5
clf = KNeighborsClassifier(n_neighbors=n_neighbors)
clf.fit(X_train,y_train)
print("KNN (k=%d)"%n_neighbors,"\t : train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train,y_train)*100 , clf.score(X_test,y_test)*100 ))
"""

########################################
######## Bayésien naïf gaussien ########

# Apprentissage : bayésien naïf gaussien
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train,y_train)
print("BNG :\t train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train,y_train)*100 , clf.score(X_test,y_test)*100 ))

########################################
######## Bayésien naïf Bernoulli ########

# Apprentissage : bayésien naïf Bernoulli
from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB()
clf.fit(X_train,y_train)
print("BNB :\t train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train,y_train)*100 , clf.score(X_test,y_test)*100 ))

########################################
######## Bayésien naïf Multinomial ########
"""
# Apprentissage : bayésien naïf multinomial
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train,y_train)
#y_validation = clf.predict(X_test)
print("BNM :\t train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train,y_train)*100 , clf.score(X_test,y_test)*100 ))

J'ai enlevé BNM car ça donnait une erreur :

Traceback (most recent call last):
  File "main.py", line 73, in <module>
    clf.fit(X_train,y_train)
  File "/usr/local/lib/python3.7/site-packages/sklearn/naive_bayes.py", line 613, in fit
    self._count(X, Y)
  File "/usr/local/lib/python3.7/site-packages/sklearn/naive_bayes.py", line 720, in _count
    raise ValueError("Input X must be non-negative")
ValueError: Input X must be non-negative

"""
########################################
################## SVM #################

# Apprentissage : SVM
from sklearn import svm
clf = svm.SVC(kernel=kernel,gamma=SVC_gamma, C=SVC_C)
clf.fit(X_train,y_train)
print("SVM :\t train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train,y_train)*100 , clf.score(X_test,y_test)*100 ))

##############################################
####### Régression logistique lbfgs ##########

# Apprentissage : régression logistique avec solveur 'lbfgs'
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(solver='lbfgs',multi_class='auto',max_iter=100) # il faut mettre 4000 pour que ça converge, mais c'est long à calculer
clf.fit(X_train,y_train)
print("lbf :\t train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train,y_train)*100 , clf.score(X_test,y_test)*100 ))

##################################################
####### Régression logistique liblinear ##########

# Apprentissage : régression logistique avec solveur 'liblinear'
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(solver='liblinear',multi_class='auto',max_iter=100)
clf.fit(X_train,y_train)
print("lib :\t train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train,y_train)*100 , clf.score(X_test,y_test)*100 ))

##################################################
########### Random Forest Classifier #############

# Apprentissage : Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train,y_train)
print("RFC :\t train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train,y_train)*100 , clf.score(X_test,y_test)*100 ))

##################################################
################# Perceptron #####################

# Apprentissage : Perceptron
from sklearn.linear_model import Perceptron
clf = Perceptron()
clf.fit(X_train,y_train)
print("Per :\t train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train,y_train)*100 , clf.score(X_test,y_test)*100 ))

##################################################
################ SGDClassifier (SGDC) ###################

# Apprentissage : SGDClassifier, descente de gradient stochastique, version classificateur
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier()
clf.fit(X_train,y_train)
print("SGD :\t train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train,y_train)*100 , clf.score(X_test,y_test)*100 ))

##################################################
####### Decision Tree Classifier (DTC) ###########

# Apprentissage : DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
print("DTC :\t train : {0:.1f}%\t test : {1:.1f}%".format( clf.score(X_train,y_train)*100 , clf.score(X_test,y_test)*100 ))


####################################################################################################
####################################################################################################

"""

Prédictions (pour 4ième seconde, non ajusté) :

KNN :	 train : 100.0%	 test : 79.8% <---
BNG :	 train : 35.6%	 test : 30.7%
BNB :	 train : 44.8%	 test : 41.2%
SVM :	 train : 86.8%	 test : 80.5% <---
lbf :	 train : 77.8%	 test : 70.8%
lib :	 train : 82.6%	 test : 73.0%
RFC :	 train : 98.6%	 test : 74.9%
Per :	 train : 43.2%	 test : 47.9%
SGD :	 train : 68.3%	 test : 61.0%
DTC :	 train : 100.0%	 test : 64.8%

Prédictions (pour 12ième seconde, ajusté) :

KNN :  train : 100.0%  test : 75.7% <---
BNG :  train : 40.6%   test : 42.5%
BNB :  train : 45.7%   test : 50.4%
SVM :  train : 86.8%   test : 81.0% <---
lbf :  train : 82.4%   test : 76.9%
lib :  train : 85.8%   test : 75.0%
RFC :  train : 98.8%   test : 78.4%
Per :  train : 65.7%   test : 58.2%
SGD :  train : 66.9%   test : 59.3%
DTC :  train : 100.0%  test : 69.8%

Tout est plus haut (sauf KNN)

"""


####################################################################################################
####################################################################################################

# Ici je vais faire une matrice de confusion
# Colonne est la réalité, ligne est la prédiction

from sklearn.metrics import confusion_matrix
clf = svm.SVC(kernel=kernel,gamma=SVC_gamma, C=SVC_C)
clf.fit(X_train,y_train)
y_true = y_test
y_pred = clf.predict(X_test)
matrix = confusion_matrix(y_true, y_pred, labels=[0,1,2])
print("\nMatrice de confusion pour SVM :")
print(matrix)
print("\nNombre de prédictions : ",len(y_true))

"""
Matrice de confusion (i,j) = (réalité,prédit) :

0 : Classique
1 : Électronique
2 : Métal

Pour 4ième seconde, non ajusté :

KNN (succès = 79.8%) :
[[95  3  3]
 [ 7 53 15]
 [12 14 65]]

SVM (succès = 80.5%) :
 [[91  7  3]
 [ 8 58  9]
 [15 10 66]]


Pour 12ième seconde, ajusté :
SVM (succès = 81%) :
[[92  6  3]
 [11 54 11]
 [11  9 71]]
Nombre de prédictions :  268
Nombre d'erreurs = 51

"""




####################################################################################################
####################################################################################################

# Ici je trouve quelles chansons sont mal prédites

nombre_derreurs = 0
erreurs = []
import math
for i in range(len(y_pred)):
	if y_true[i]!=y_pred[i]:
		nombre_derreurs += 1
		erreurs.append([int(id_test[i]),y_true[i],y_pred[i]])
		#print("i : ",i, "\ty_true : ",y_true[i],"\ty_pred : ",y_pred[i],"id_test =",int(id_test[i]))
print("Nombre d'erreurs =",nombre_derreurs)
#print(erreurs)
df_erreurs = pd.DataFrame(erreurs,columns=["id_test","y_true","y_pred"])
df_erreurs.sort_values(by=['id_test'], inplace=True)
id_erreurs = list(df_erreurs['id_test'])
print("\n",df_erreurs)


"""
import matplotlib.pyplot as plt
#plt.plot(id_erreurs)
binwidth = 30
plt.hist(id_erreurs,bins=range(min(id_erreurs), max(id_erreurs) + binwidth, binwidth))
plt.show()
"""


"""

Pour 4ième seconde, non ajusté :

Nombre d'erreurs = 52
    id_test  y_true  y_pred
42       25     0.0     1.0
11      169     0.0     1.0
23      223     0.0     1.0
26      315     0.0     1.0
44      316     0.0     2.0
15      322     0.0     1.0
20      334     0.0     2.0
16      340     0.0     2.0
18      343     0.0     1.0
14      348     0.0     1.0
38      365     1.0     2.0
17      400     1.0     0.0
1       442     1.0     2.0
28      451     1.0     0.0
51      475     1.0     0.0
2       501     1.0     0.0
46      533     1.0     2.0
19      550     1.0     2.0
34      587     1.0     0.0
31      588     1.0     2.0
0       600     1.0     2.0
13      608     1.0     2.0
45      628     1.0     2.0
30      647     1.0     0.0
7       649     1.0     0.0
24      650     1.0     0.0
40      701     1.0     2.0
50      723     2.0     0.0
25      724     2.0     0.0
5       725     2.0     0.0
22      742     2.0     0.0
3       769     2.0     1.0
21      775     2.0     0.0
6       777     2.0     0.0
47      780     2.0     0.0
4       781     2.0     0.0
43      785     2.0     0.0
39      789     2.0     0.0
36      798     2.0     0.0
37      799     2.0     0.0
8       802     2.0     0.0
9       808     2.0     0.0
29      830     2.0     1.0
41      838     2.0     1.0
33      856     2.0     1.0
27      863     2.0     1.0
48      908     2.0     0.0
49      941     2.0     1.0
35      956     2.0     1.0
32      968     2.0     1.0
12     1008     2.0     1.0
10     1023     2.0     1.0


Pour 12ième seconde, ajusté :

     id_test  y_true  y_pred
0        19     0.0     2.0
31       39     0.0     2.0
44      100     0.0     1.0
27      223     0.0     1.0
8       307     0.0     1.0
42      310     0.0     1.0
30      315     0.0     1.0
24      334     0.0     1.0
22      343     0.0     2.0
14      360     1.0     2.0
41      365     1.0     2.0
19      379     1.0     0.0
10      399     1.0     0.0
21      400     1.0     0.0
3       429     1.0     2.0
2       442     1.0     2.0
32      451     1.0     0.0
4       501     1.0     2.0
33      517     1.0     2.0
47      533     1.0     0.0
23      550     1.0     2.0
18      554     1.0     2.0
37      587     1.0     0.0
20      608     1.0     2.0
35      624     1.0     0.0
45      633     1.0     2.0
34      647     1.0     0.0
13      649     1.0     0.0
28      650     1.0     0.0
1       656     1.0     2.0
46      697     1.0     0.0
49      723     2.0     0.0
29      724     2.0     0.0
9       725     2.0     0.0
50      756     2.0     1.0
5       760     2.0     1.0
17      763     2.0     1.0
16      768     2.0     1.0
11      777     2.0     0.0
48      780     2.0     0.0
7       781     2.0     0.0
43      786     2.0     1.0
40      799     2.0     0.0
15      802     2.0     0.0
39      835     2.0     0.0
12      846     2.0     0.0
36      859     2.0     1.0
6       861     2.0     1.0
25      919     2.0     1.0
26      994     2.0     0.0
38     1024     2.0     1.0


"""




####################################################################################################
####################################################################################################

# On peut faire des plot et scatter de points

"""
# Isomap : Isometric mapping
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
iso = Isomap(n_neighbors=1, n_components=2)
proj = iso.fit_transform(X)
plt.scatter(proj[:,0],proj[:,1],c=y)
plt.colorbar()
plt.show()
"""


# PCA : Principal Component Analysis
import matplotlib.pylab as plt
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
proj = pca.fit_transform(X)
plt.scatter(proj[:,0],proj[:,1], c=y)
plt.colorbar()
# On ne regarde pas les points qui sont trop loin
plt.xlim(-32, 20)
plt.ylim(-11, 21)
plt.title('Analyse en composantes principales (PCA)',size=16);
plt.show()



####################################################################################################
####################################################################################################

# Les prédictions sur le vaporwave

# On importe le spectre du vaporwave
Xy_vaporwave = (pd.read_csv(data_path + "/data_vaporwave.csv",sep = ',',skipinitialspace=False)).to_numpy()
# On découpe les données en features et target
id_data_vaporwave = Xy_vaporwave[:,0] 
y_vaporwave       = Xy_vaporwave[:,1]  # le target : genre = 3
X_vaporwave       = Xy_vaporwave[:,2:] # features : spectre

clf = svm.SVC(kernel=kernel,gamma=SVC_gamma, C=SVC_C)
clf.fit(X,y) # on entraîne le modèle sur tout le classique, l'électronique et le métal
y_true = y_vaporwave
y_pred = clf.predict(X_vaporwave)
print("\nNombre de prédictions (pour le vaporwave) : ",len(y_pred))

nombre_derreurs = 0
erreurs = []
count_classique = 0
count_electronique = 0
count_metal = 0
for i in range(len(y_pred)):
  if y_pred[i]==0:count_classique+=1
  if y_pred[i]==1:count_electronique+=1
  if y_pred[i]==2:count_metal+=1
  erreurs.append([int(id_data_vaporwave[i]),y_true[i],y_pred[i]])
#print(erreurs)
print("Prédictions classique :\t\t",count_classique,"\tPourcentage :\t",100*count_classique/len(y_pred),"%")
print("Prédictions électronique :\t",count_electronique,"\tPourcentage :\t",100*count_electronique/len(y_pred),"%")
print("Prédictions métal :\t\t",count_metal,"\tPourcentage :\t",100*count_metal/len(y_pred),"%")
df_erreurs = pd.DataFrame(erreurs,columns=["id_test","y_true","y_pred"])
df_erreurs.sort_values(by=['id_test'], inplace=True)
id_erreurs = list(df_erreurs['id_test'])
print("\n",df_erreurs)

"""

Prédictions sur le vaporwave

Pour la 4ième seconde, non ajusté :

14.9% du classique
55.3% de l'électronique
29.8% du métal


Pour la 12ième seconde, ajusté :

Nombre de prédictions (pour le vaporwave) :  47
Prédictions classique :     9  Pourcentage :  19.148936170212767 %
Prédictions électronique : 16  Pourcentage :  34.04255319148936 %
Prédictions métal :        22  Pourcentage :  46.808510638297875 %

     id_test  y_true  y_pred
0         1     3.0     2.0
1         2     3.0     2.0
2         3     3.0     2.0
3         4     3.0     0.0
4         5     3.0     2.0
5         6     3.0     1.0
6         7     3.0     2.0
7         8     3.0     2.0
8         9     3.0     2.0
9        10     3.0     2.0
10       11     3.0     2.0
11       12     3.0     2.0
12       13     3.0     2.0
13       14     3.0     2.0
14       15     3.0     1.0
15       16     3.0     2.0
16       17     3.0     1.0
17       18     3.0     1.0
18       19     3.0     1.0
19       20     3.0     1.0
20       21     3.0     1.0
21       22     3.0     0.0
22       23     3.0     0.0
23       24     3.0     2.0
24       25     3.0     0.0
25       26     3.0     1.0
26       27     3.0     1.0
27       28     3.0     0.0
28       29     3.0     1.0
29       30     3.0     2.0
30       31     3.0     2.0
31       32     3.0     1.0
32       33     3.0     2.0
33       34     3.0     2.0
34       35     3.0     0.0
35       36     3.0     2.0
36       37     3.0     1.0
37       38     3.0     1.0
38       39     3.0     0.0
39       40     3.0     2.0
40       41     3.0     0.0
41       42     3.0     0.0
42       43     3.0     1.0
43       44     3.0     1.0
44       45     3.0     1.0
45       46     3.0     2.0
46       47     3.0     2.0


Bref, pour la 4ième seconde, non ajusté :

14.9% classique (7/47)
55.3% électronique (26/47)
29.8% métal (14/47)

Pour la 12ième seconde, ajusté :

19.1% classique (9/47)
34.0% électronique (16/47)
46.8% métal (22/47)

Si on fait une moyenne entre les deux :

17.0% classique
44.7% électronique
38.3% métal

"""














