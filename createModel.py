from sklearn_deltatfidf import DeltaTfidfVectorizer

from project.Preprocess import preProcess


def deltaMatrix(stemmedList,score):
    v = DeltaTfidfVectorizer()
    v.ngram_range = (2, 2)
    result = v.fit_transform(stemmedList, score)

    return result

def main():
    p = preProcess()

    stemmedList, score = p.main()

    tfidfMatrix = deltaMatrix(stemmedList,score)

    print(tfidfMatrix.shape)

main()

