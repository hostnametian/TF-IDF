import numpy as np
import pandas as pd
import math
from sklearn.metrics.pairwise import cosine_similarity

#1.声明文档 分词 去重合并
Q = 'gold silver truck gold'
D1 = 'Shipment of gold damaged in a fire'
D2 = 'Delivery of silver arrived well in a silver truck gold'
D3 = 'Shipment of gold all arrived in a truck well'

split1 = D1.split(' ')
split2 = D2.split(' ')
split3 = D3.split(' ')
wordSet = set(split1).union(split2, split3)  #通过set去重来构建词库


#2.统计词项tf在文档Di中出现的次数，也就是词频。
def computeTF(wordSet, split):
    tf = dict.fromkeys(wordSet, 0)
    for word in split:
        tf[word] += 1
    return tf


split_q = Q.split(' ')  #分词
tf_q = computeTF(wordSet, split_q)  #计算Q的词频
tf1 = computeTF(wordSet, split1)
tf2 = computeTF(wordSet, split2)
tf3 = computeTF(wordSet, split3)
print('tf1:\n', tf1)

print('tf2:\n', tf2)

print('tf_q:\n', tf_q)

print('tf3:\n', tf3)


#3.计算逆文档频率IDF
def computeIDF(tfList):
    idfDict = dict.fromkeys(tfList[0], 0)  #词为key，初始值为0
    N = len(tfList)  #总文档数量
    for tf in tfList:  # 遍历字典中每一篇文章
        for word, count in tf.items():  #遍历当前文章的每一个词
            if count > 0:  #当前遍历的词语在当前遍历到的文章中出现
                idfDict[word] += 1  #包含词项tf的文档的篇数df+1
    for word, Ni in idfDict.items():  #利用公式将df替换为逆文档频率idf
        idfDict[word] = math.log10(N / Ni)  #N,Ni均不会为0
    return idfDict  #返回逆文档频率IDF字典


idfs = computeIDF([tf1, tf2, tf3])
print('idfs:\n', idfs)

#4.计算tf-idf(term frequency–inverse document frequency)


def computeTFIDF(tf, idfs):  #tf词频,idf逆文档频率
    tfidf = {}
    for word, tfval in tf.items():
        tfidf[word] = tfval * idfs[word]
    return tfidf


tfidf1 = computeTFIDF(tf1, idfs)
tfidf2 = computeTFIDF(tf2, idfs)
tfidf3 = computeTFIDF(tf3, idfs)
tfidf = pd.DataFrame([tfidf1, tfidf2, tfidf3])
#5.查询与文档Q最相似的文章

tfidf_q = computeTFIDF(tf_q, idfs)  #计算Q的tf_idf(构建向量)
ans = pd.DataFrame([tfidf1, tfidf2, tfidf3, tfidf_q])
print('TF-IDF(tfidf1, tfidf2, tfidf3, tfidf_q):\n', ans)

print("-----------------------------------------------------------\n")
#6.计算Q和文档D1,D2,D3的相似度（向量的内积）
print('Similarity of vector inner product（The bigger the better）:')
print('文档D1和文档Q的相似度SC(Q, D1) :', (ans.loc[0, :] * ans.loc[3, :]).sum())
print('文档D2和文档Q的相似度SC(Q, D2) :', (ans.loc[1, :] * ans.loc[3, :]).sum())
print('文档D3和文档Q的相似度SC(Q, D3) :', (ans.loc[2, :] * ans.loc[3, :]).sum())

print("-----------------------------------------------------------\n")

#7.计算Q和文档D1,D2,D3的相似度（余弦相似度）
print("Cosine similarity:(The bigger the better)")
# 余弦值越接近1，就表明夹角越接近0度，也就是两个向量越相似，这就叫余弦相似性
Q_vector = list(ans.loc[3, :])
D1_vector = list(ans.loc[0, :])
D2_vector = list(ans.loc[1, :])
D3_vector = list(ans.loc[2, :])
# print("Q_vector\n",Q)
# print("D1_vector\n",D1)
# print("D2_vector\n",D2)
# print("D3_vector\n",D3)
s1 = cosine_similarity([Q_vector], [D1_vector])
s2 = cosine_similarity([Q_vector], [D2_vector])
s3 = cosine_similarity([Q_vector], [D3_vector])
print("文档D1与文档Q之间的余弦相似度为：", s1)
print("文档D2与文档Q之间的余弦相似度为：", s2)
print("文档D3与文档Q之间的余弦相似度为：", s3)

print("-----------------------------------------------------------\n")
# 8.计算Q和文档D1,D2,D3的欧氏距离（Euclidean distance）
print("Euclidean distance:(The smaller the better)")
Q_E = np.array(Q_vector)
D1_E = np.array(D1_vector)
D2_E = np.array(D2_vector)
D3_E = np.array(D3_vector)

dist_D1_E = np.sqrt(np.sum(np.square(D1_E - Q_E)))
dist_D2_E = np.sqrt(np.sum(np.square(D2_E - Q_E)))
dist_D3_E = np.sqrt(np.sum(np.square(D3_E - Q_E)))
print("文档D1与文档Q之间的欧式距离为：", dist_D1_E)
print("文档D2与文档Q之间的欧式距离为：", dist_D2_E)
print("文档D3与文档Q之间的欧式距离为：", dist_D3_E)

print("-----------------------------------------------------------\n")

# 8.计算Q和文档D1,D2,D3的Jaccard相似度（Jaccard Similarity）
print("Jaccard Similarity:(The bigger the better)")
Q_J = set(Q_vector)
D1_J = set(D1_vector)
D2_J = set(D2_vector)
D3_J = set(D3_vector)

# print(Q_J)
# print(D1_J)
# print(D2_J)
# print(D3_J)
# print(len(Q_J & D1_J))


j1=(len(Q_J & D1_J))/(len(Q_J) + len(D1_J)-len(Q_J & D1_J))
j2=(len(Q_J & D2_J))/(len(Q_J) + len(D2_J)-len(Q_J & D2_J))
j3=(len(Q_J & D3_J))/(len(Q_J) + len(D3_J)-len(Q_J & D3_J))

print("文档D1与文档Q之间的Jaccard相似度：",j1)
print("文档D2与文档Q之间的Jaccard相似度：",j2)
print("文档D3与文档Q之间的Jaccard相似度：",j3)