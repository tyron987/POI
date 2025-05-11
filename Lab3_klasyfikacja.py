import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA

dataset = input("Podaj ścieżkę do pliku z kategoriami tekstur: ").strip()
features = pd.read_csv(dataset, sep=',')

data = np.array(features)
X = (data[:,:-1]).astype('float64') # dane wejściowe
Y = data[:,-1] # etykiety klas

x_transform = PCA(n_components=3) # transformowanie danych do przestrzeni trójwymiarowej

Xt = x_transform.fit_transform(X)

red = Y == 'drzwi_1'
green = Y == 'drzwi_2'
orange = Y == 'drzwi_3'
magenta = Y == 'drzwi_4'
yellow = Y == 'panele_1'
blue = Y == 'panele_2'
pink = Y == 'panele_3'
cyan = Y == 'płytka_1'
purple = Y == 'płytka_2'
brown = Y == 'płytka_3'
rose = Y == 'płytka_4'
grey = Y == 'tynk_1'
black = Y == 'tynk_2'
silver = Y == 'tynk_3'
emerald = Y == 'tynk_4'
turquoise = Y == 'ściana_1'
teal = Y == 'ściana_2'
aqua = Y == 'ściana_3'

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Xt[red, 0], Xt[red, 1], Xt[red, 2], c="r", label="drzwi_1")
ax.scatter(Xt[green, 0], Xt[green, 1], Xt[green, 2], c="g", label="drzwi_2")
ax.scatter(Xt[orange, 0], Xt[orange, 1], Xt[orange, 2], c="orange", label="drzwi_3")
ax.scatter(Xt[magenta, 0], Xt[magenta, 1], Xt[magenta, 2], c="m", label="drzwi_4")
ax.scatter(Xt[yellow, 0], Xt[yellow, 1], Xt[yellow, 2], c="y", label="panele_1")
ax.scatter(Xt[blue, 0], Xt[blue, 1], Xt[blue, 2], c="b", label="panele_2")
ax.scatter(Xt[pink, 0], Xt[pink, 1], Xt[pink, 2], c="pink", label="panele_3")
ax.scatter(Xt[cyan, 0], Xt[cyan, 1], Xt[cyan, 2], c="c", label="płytka_1")
ax.scatter(Xt[purple, 0], Xt[purple, 1], Xt[purple, 2], c="purple", label="płytka_2")
ax.scatter(Xt[brown, 0], Xt[brown, 1], Xt[brown, 2], c="brown", label="płytka_3")
ax.scatter(Xt[rose, 0], Xt[rose, 1], Xt[rose, 2], c="rosybrown", label="płytka_4")
ax.scatter(Xt[grey, 0], Xt[grey, 1], Xt[grey, 2], c="grey", label="tynk_1")
ax.scatter(Xt[black, 0], Xt[black, 1], Xt[black, 2], c="black", label="tynk_2")
ax.scatter(Xt[silver, 0], Xt[silver, 1], Xt[silver, 2], c="silver", label="tynk_3")
ax.scatter(Xt[emerald, 0], Xt[emerald, 1], Xt[emerald, 2], c="lime", label="tynk_4")
ax.scatter(Xt[turquoise, 0], Xt[turquoise, 1], Xt[turquoise, 2], c="turquoise", label="ściana_1")
ax.scatter(Xt[teal, 0], Xt[teal, 1], Xt[teal, 2], c="teal", label="ściana_2")
ax.scatter(Xt[aqua, 0], Xt[aqua, 1], Xt[aqua, 2], c="aqua", label="ściana_3")

ax.legend(loc='upper right', fontsize='small')

classifier = svm.SVC(gamma='auto') # klasyfikator SVM

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33) # dane podzielone są na
# zbiór treningowy (1/3) i zbiór testowy (2/3)

classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print("Dokładność klasyfikatora wynosi: " + str(acc) + "\n")

cm = confusion_matrix(y_test, y_pred, normalize='true')

print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['drzwi_1','drzwi_2','drzwi_3','drzwi_4','panele_1',
                                                                   'panele_2','panele_3','płytka_1','płytka_2',
                                                                   'płytka_3','płytka_4','tynk_1','tynk_2','tynk_3',
                                                                   'tynk_4','ściana_1','ściana_2','ściana_3'])
disp.plot(cmap='Blues')
plt.show()
