from numpy import zeros
from scipy.linalg import svd
from math import log  # needed for TFIDF
from numpy import asarray, sum

import codecs
import pymorphy2
import matplotlib.pyplot as plt

titles = []

#file = codecs.open( "text.txt", "r", "utf_8_sig") # файл
#titles = file.read()  # считывание построчно
#print(titles),
#file.close()`

# считываем с файла слова, которые необходимо исключить
stopwords = []
with codecs.open("stopwords.txt", "r", "utf_8_sig") as f:
    stopwords = f.read().splitlines()

# считываем с файла обрабатываемый текст
titles = []
for document in codecs.open('text.txt', 'r', 'utf-8'):
    titles.append(document)

# titles = [
#    "Британская: полиция знает о местонахождении основателя WikiLeaks",
#    "В суде США начинается процесс против россиянина, рассылавшего спам",
#    "Церемонию вручения Нобелевской премии мира бойкотируют 19 стран",
#    "В Великобритании арестован основатель сайта Wikileaks Джулиан Ассандж",
#    "Украина игнорирует церемонию вручения Нобелевской премии",
#    "Шведский суд отказался рассматривать апелляцию основателя Wikileaks",
#    "НАТО и США разработали планы обороны стран Балтии против России",
#    "Полиция Великобритании нашла основателя WikiLeaks, но, не арестовала",
#    "В Стокгольме и Осло сегодня состоится вручение Нобелевских премий"
# ]

ignorechars = ''',:'!\t'''
#print(ignorechars)

class LSA(object):
    def __init__(self, stopwords, ignorechars):
        self.stopwords = stopwords
        self.ignorechars = ignorechars
        self.wdict = {}
        self.dcount = 0

    def norm(self, x):
        morph = pymorphy2.MorphAnalyzer()
        p = morph.parse(x)[0].normal_form
        return p

    def parse(self, doc):
        words = doc.split();
        for w in words:
            # на доработку
            w = w.lower().replace(",", "")
            w = w.lower().replace(".", "")
            w = w.lower().replace("!", "")
            w = w.lower().replace("?", "")
            w = w.lower().replace(":", "")
            w = w.lower().replace(";", "")
            #w = w.lower().translate(self.ignorechars)
            w = self.norm(w)
            if w in self.stopwords:
                continue
            elif w in self.wdict:
                self.wdict[w].append(self.dcount)
            else:
                self.wdict[w] = [self.dcount]
        self.dcount += 1

    # rows -> keywords (occur more than twice), cols -> documentID
    def build(self):

        self.keys = [k for k in self.wdict.keys() if len(self.wdict[k]) > 1] #убираем одиночные слова
        self.keys.sort()
        self.A = zeros([len(self.keys), self.dcount])
        for i, k in enumerate(self.keys):
            for d in self.wdict[k]:
                self.A[i, d] += 1

    def calc(self):
        self.U, self.S, self.Vt = svd(self.A)


    def TFIDF(self):
        WordsPerDoc = sum(self.A, axis=0)
        DocsPerWord = sum(asarray(self.A > 0, 'i'), axis=1)
        rows, cols = self.A.shape
        for i in range(rows):
            for j in range(cols):
                self.A[i, j] = (self.A[i, j] / WordsPerDoc[j]) * log(float(cols) / DocsPerWord[i])


    def printA(self):
        print('Here is the count matrix')
        print(self.A)

    def printSVD(self):
        file_S = open("result-S.txt", "w")  # output "w" - переписать, "a" - дописать в конец.
        print('the singular values S-matrix')
        print(self.S)
        file_S.write('S-matrix\n')
        i = 0
        while i < 1:
            file_S.write(str(self.S))
            i += 1
        file_S.close

        file_U = open("result-U.txt", "w")  # output "w" - переписать, "a" - дописать в конец.
        print('the columns of the U-matrix')
        print(- 1 * self.U)
        file_U.write('U-matrix\n')
        i = 0
        j = 0
        for i in range(len(self.U)):
            for j in range(len(self.U[i])):
                file_U.write(str(' '))
                file_U.write(str(-1 * self.U[i][j])) #wtf
                j+=1
            i+=1
            file_U.write('\n')
        file_U.close

        file_Vt = open("result-Vt.txt", "w")  # output "w" - переписать, "a" - дописать в конец.
        print('the rows of the Vt-matrix')
        print(- 1 * self.Vt)
        file_Vt.write('Vt-matrix\n')
        i = 0
        j = 0
        for i in range(len(self.Vt)):
            for j in range(len(self.Vt[i])):
                file_Vt.write(str(' '))
                file_Vt.write(str(-1 * self.Vt[i][j]))  # wtf
                j += 1
            i += 1
            file_Vt.write('\n')
        file_Vt.close


    def TFIDF(self):
        WordsPerDoc = sum(self.A, axis=0)
        DocsPerWord = sum(asarray(self.A > 0, 'i'), axis=1)
        rows, cols = self.A.shape
        for i in range(rows):
            for j in range(cols):
                self.A[i, j] = (self.A[i, j] / WordsPerDoc[j]) * log(float(cols) / DocsPerWord[i])

    def graph(self):
        plt.scatter(self.U[:, 0], self.U[:, 1])
        for i in range(len(self.keys)):
            plt.annotate(s=self.keys[i], xy=(self.U[i, 0], self.U[i, 1]))
        plt.show()


    @staticmethod
    def main() -> object:
        mylsa = LSA(stopwords, ignorechars)
        for t in titles:
            mylsa.parse(t)
        mylsa.build()
        mylsa.printA()
        mylsa.calc()
        mylsa.printSVD()
        mylsa.graph()

if __name__ == '__main__':
    LSA.main()
