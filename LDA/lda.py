# -*- coding: utf-8 -*-
'''
@author: josephqian
@contact: jianpeiqian1990@gmail.com
@software: jupyter
@file: ttm.py
@initialtime:  4/3/20 14:33 PM
'''
# log
'''
@version: v1.1-202003141100
@modification:
    修改后与ttm模型结构上保持一致

import random
import numpy as np
import pandas as pd
from gensim import corpora
import datetime
import matplotlib.pyplot as plt
import scipy.stats
'''

class LdaModel:
    #主模型初始化
    def __init__(self, data, dictionary, coef_alpha, beta, NumberOfIterations,  K):
        #模型变量定义
        self.alpha = coef_alpha/K
        self.beta = beta
        self.NumberOfIterations = NumberOfIterations
        self.K = K
        self.docs=data
        self.M = len(self.docs)
        self.dictionary = dictionary
        self.V = len(self.dictionary)
        self.likelihoodList = [] * self.NumberOfIterations
        #计数变量定义
        self.z = [[]for i in range(self.M)]
        self.nw = [[0]*self.V for _ in range(self.K)]
        self.nd = [[0]*self.K for _ in range(self.M)]
        self.nwsum = [0]*self.K
        self.ndsum = [0]*self.M      
    
        #模型计数向量和计数矩阵初始化 
        print("1. Begin to initialize the LDA model...")  
        for m in range(self.M):
            N = len(self.docs[m])  
            for n in range(N):  
                self.z[m].append(0)
            # initialize for z
            for n in range(N):  
                k = random.randint(0, self.K-1)
                self.z[m][n] = k             
                w=self.docs[m][n]
                wid=self.dictionary.token2id[w]
                self.nw[k][wid] += 1
                self.nd[m][k] += 1
                self.nwsum[k] += 1
            self.ndsum[m] = N            
        print(">>> Topic assignment done!!")
        
    #用与测试数据时调用，进行初始化
    def inf_init(self):
        self.new_z = [[]for i in range(self.new_M)]
        self.new_nd = [[0]*self.K for _ in range(self.new_M)]
        self.new_ndsum = [0]*self.new_M   
        for m in range(self.new_M):
            N = len(self.new_docs[m]) 
            for n in range(N): 
                self.new_z[m].append(0)
            for n in range(N): 
                k = random.randint(0, self.K-1)
                self.new_z[m][n] = k              
                self.new_nd[m][k] += 1
            self.new_ndsum[m] = N
    
    def utils_ComputeTransProb(self, m, n):
        w = self.docs[m][n]
        wid = self.dictionary.token2id[w]
        p = [0] * self.K
        Kalpha = self.K * self.alpha
        Vbeta = self.V * self.beta
        for k in range(self.K):
            p[k] = (float(self.nw[k][wid]) + self.beta) / (float(self.nwsum[k]) + Vbeta) * (float(self.nd[m][k]) + self.alpha)
        return p
    
    def utils_MultiSample(self, ProbList):
        size = len(ProbList)
        #计算累计概率密度函数
        for i in range(size-1):
            ProbList[i+1] += ProbList[i]
        #随机产生一个［0，1）的小数
        u = random.random()
        res = 0
        for k in range(size):
            if ProbList[k] >= u * ProbList[size - 1]:
                #抽样结果
                res = k
                break
        #res为抽样后的主题编号
        return res
    
    def output_ComputeTheta(self):
        theta = [[0]*self.K for _ in range(self.M)]
        #提前计算Kalpha有助于减少循环中的乘法运算次数
        Kalpha = self.K * self.alpha
        for m in range(self.M):
            for k in range(self.K):
                theta[m][k] = (float(self.nd[m][k]) + self.alpha) / (float(self.ndsum[m]) + Kalpha)
        return theta
        
    def output_ComputePhi(self):
        phi = [[0]*self.V for _ in range(self.K)]
        Vbeta = self.V * self.beta
        for k in range(self.K):
            for w in self.dictionary:
                #wid = self.dictionary.token2id[w]
                phi[k][w] = (float(self.nw[k][w]) + self.beta) / (float(self.nwsum[k]) + Vbeta)
        return phi
    
    def inf_ComputeTheta(self):
        new_theta = [[0]*self.K for _ in range(self.new_M)]
        Kalpha = self.K * self.alpha
        for m in range(self.new_M):
            for k in range(self.K):
                new_theta[m][k] = (float(self.new_nd[m][k]) + self.alpha) / (float(self.new_ndsum[m]) + Kalpha)
        return new_theta
    
    #训练数据的Gibbs采样算法
    def gibbs_sampling(self,m,n):
        k = self.z[m][n]
        #step1:对应位置上的词的ID
        w = self.docs[m][n]
        wid = self.dictionary.token2id[w]
        self.nw[k][wid] -= 1
        self.nd[m][k] -= 1
        self.nwsum[k] -= 1
        #step2:完全条件概率计算，应用Gibbs采样公式得到每个词的主题选择概率
        p = self.utils_ComputeTransProb(m, n)
        #Gibbs采用公式得到的结果是正比关系非等式关系，因此p需要归一化，
        p_norm=list(map(lambda x:x*(1/sum(p)),p))       
        #step3:多项式分布采样/更新主题
        k_new = self.utils_MultiSample(p_norm)      
        #step4:更新计数
        self.nw[k_new][wid] += 1
        self.nd[m][k_new] += 1
        self.nwsum[k_new] += 1
        return k_new
           
    #测试数据的Gibbs采样算法
    def inf_gibbs_sampling(self, m, n, phi):
        #phi =  self.output_ComputePhi()
        #获取要更新主题的位置（m，n）上对应的词wid
        k = self.new_z[m][n]
        w = self.new_docs[m][n]
        wid = self.dictionary.token2id[w]
        #词wid对应的计数减1
        self.new_nd[m][k] -= 1
        #计算wid的主题分布（完全条件概率公式）       
        p = [0]*self.K
        for _k in range(self.K):            
            p[_k] = phi[_k][wid] * (float(self.new_nd[m][_k]) + self.alpha)
        p_norm=list(map(lambda x:x*(1/sum(p)),p))
        #根据上一分布采样wid新的主题编号
        k_new = self.utils_MultiSample(p_norm)
        self.new_nd[m][k_new] += 1
        return k_new
    
    #调用该函数估计LDA模型参数
    def estimate(self):
        print("2. Begin to estimate the LDA model...")
        starttime = datetime.datetime.now()        
        for i in range(1, self.NumberOfIterations + 1):
            for m in range(self.M):
                N = len(self.docs[m])
                for n in range(N):
                    k_new = self.gibbs_sampling(m, n)
                    #为当前词分派新主题
                    self.z[m][n] = k_new
            #每对语料库循环一次gibbs采样就计算一次似然
            theta = np.array(self.output_ComputeTheta())                #增加，先计算theta
            phi = np.array(self.output_ComputePhi())                   #增加，先计算phi
            log_likelihood = self.likelihood(theta=theta, phi=phi)         #参数中增加theta和phi
            self.likelihoodList.append(log_likelihood)                      
        endtime = datetime.datetime.now()
        print('>>> runtime =',(endtime - starttime).seconds,'s')
        print(">>> LDA model estimated!!")
        #模型迭代训练结果输出
        self.plt_likelihood() 

    #调用该函数估计测试数据/新数据的参数
    def inf_estimate(self, newdata, NewIterations=None):
        print("5. Begin to test the LDA model...")
        starttime = datetime.datetime.now()
        if NewIterations == None:
            NumberOfIterations = self.NumberOfIterations
        else:
            NumberOfIterations = NewIterations
        self.new_likelihoodList = [] * NumberOfIterations               #inf循环次数改变时，增加该list，完成likelihood计算与绘        
        self.new_docs = newdata
        self.new_M = len(self.new_docs)
        self.inf_init()
        #to estimate the topic structure of unseen documents (querying)
        phi = self.output_ComputePhi()          #增加，先计算phi
        for i in range(1, NumberOfIterations + 1):
            for m in range(self.new_M):
                N = len(self.new_docs[m])
                for n in range(N):
                    k_new = self.inf_gibbs_sampling(m, n)
                    self.new_z[m][n] = k_new
            每对语料库循环一次gibbs采样就计算一次似然
            theta = np.array(self.inf_ComputeTheta())                #增加，先计算theta
            log_likelihood = self.likelihood(theta, phi, newdata)        #参数中增加theta和phi
            self.new_likelihoodList.append(log_likelihood)  
        endtime = datetime.datetime.now()
        print('>>> runtime =',(endtime - starttime).seconds,'s')
        print(">>> LDA model tested!!","\n")  
        self.plt_likelihood(NumberOfIterations)                     #增加参数NumberOfIterations
        
    def perplexity(self, data=None):
        if data == None:
            docs = self.docs
            M = self.M
            theta = np.array(self.output_ComputeTheta())
        else:
            docs = data
            M = self.new_M
            theta = np.array(self.inf_ComputeTheta())
        
        phi = np.array(self.output_ComputePhi())
        sum_prob_d = 0
        N = 0
        for m in range(M):
            prob_d = 0
            for w in docs[m]:
                prob_w = 0
                wid = self.dictionary.token2id[w]
                for k in range(self.K):
                    prob_w += phi[k][wid] * theta[m][k]
                prob_d += np.log(prob_w)         
            sum_prob_d += prob_d
            N += len(docs[m])
        perplex = np.exp(-sum_prob_d / N)
        return perplex     
    
    def likelihood(self, theta, phi, data=None):           #增加参数theta和phi
        if data == None:
            docs = self.docs
            M = self.M
        else:
            docs = data
            M = self.new_M
        
        log_likelihood = 0
        for m in range(M):
            prob_d = 0
            for w in docs[m]:
                wid = self.dictionary.token2id[w]
                prob_w = 0
                for k in range(self.K):
                    prob_w += phi[k][wid] * theta[m][k]
                prob_d += np.log(prob_w)
            log_likelihood += prob_d
        return log_likelihood 
    
    def plt_likelihood(self, NewIterations=None):                    #区分训练与测试
        if NewIterations == None:
            NumberOfIterations = self.NumberOfIterations
            x = range(1,self.NumberOfIterations+1)
            y = self.likelihoodList
        else:
            NumberOfIterations = NewIterations
            x = range(1,NumberOfIterations+1)
            y = self.new_likelihoodList       
        
        plt.plot(x, y, color="green", linewidth=2)
        plt.xlabel("Number of Iterations")
        plt.ylabel("Log-likelihood")
        print("3. Log-likelihood curve after the iteration is...")
        plt.show()
    
    def kl_divergence(self):
        phi = self.output_ComputePhi()
        KL = 0
        count = 0
        for i in range(self.K):
            for j in range(self.K):
                if i != j:
                    count += 1
                    KL += scipy.stats.entropy(phi[i],phi[j]) 
        avg_KL = KL / count
        return avg_KL
    
    def get_doc_topic(self):
        documents = []
        topic = [] 
        probability = []
        stata_topics = np.zeros(self.K)
        
        theta = self.output_ComputeTheta()      
        for m in range(self.M):
            train_doc = str(self.docs[m])
            documents.append(train_doc)
            inf_k = np.argmax(theta[m])
            topic.append(inf_k)
            inf_prob = np.max(theta[m])
            probability.append(inf_prob)
            stata_topics[inf_k] += 1
        p_tr_docs = pd.DataFrame(index=range(self.M),columns=["documents","topic","probability"])
        p_tr_docs["documents"] = documents
        p_tr_docs["topic"] = topic
        p_tr_docs["probability"] = probability
        #p_tr_docs.to_csv('prediction_of_train_documents.csv',encoding='gbk')
        
        print("4. Topic distribution of the traning set is...")
        fig,ax=plt.subplots()
        ax.bar(range(self.K),stata_topics,color='olive', width=0.5)
        ax.set_xlabel("Index of topics")  #设置x轴标签
        ax.set_ylabel("Number of documents")  #设置y轴标签
        ax.set_title("Topic distribution of the traning set")  #设置标题
        plt.show()  #显示图像
        
        return p_tr_docs            #增加返回值
              
    #调用该函数输出：应用最优K的LDA模型主题-词分布结果
    def get_topic_term(self, k, top):
        phi = self.output_ComputePhi()
        termList = []
        probList = []
        for i,w in enumerate(phi[k]):            
            term = dict(self.dictionary.items())[i]     
            termList.append(term)
            probList.append(round(w,5))
            
        result = [[] * 2 for _ in range(self.V)]
        for t in range(self.V):
            result[t].append(termList[t])
            result[t].append(probList[t])       
        
        result_sort = sorted(result, key=lambda x: (x[1], x[0]), reverse = True)        
        termList_sort = [term[0] for term in result_sort]
        probList_sort = [prob[1] for prob in result_sort]
        
        print('>>>第',k,'个主题前',top,'个最频繁的词与概率如下：')
        print('>>>词：',termList_sort[:top])
        print('>>>概率： ',probList_sort[:top])

    #调用该函数输出：测试数据集在LDA模型上的文档-主题分布结果
    def inf_doc_topic(self):        
        documents = []
        topic = [] 
        probability = []
        stata_topics = np.zeros(self.K)
        
        new_theta = self.inf_ComputeTheta()   
        for m in range(self.new_M):
            train_doc = str(self.new_docs[m])
            documents.append(train_doc)
            inf_k = np.argmax(new_theta[m])
            topic.append(inf_k)
            inf_prob = np.max(new_theta[m])
            probability.append(inf_prob)
            stata_topics[inf_k] += 1
        p_ts_docs = pd.DataFrame(index=range(self.new_M),columns=["documents","topic","probability"])
        p_ts_docs["documents"] = documents
        p_ts_docs["topic"] = topic
        p_ts_docs["probability"] = probability
        #p_ts_docs.to_csv('prediction_of_test_documents.csv',encoding='gbk')
        
        print("6. Topic distribution of the test set is...")
        fig,ax=plt.subplots()
        ax.bar(range(self.K),stata_topics,color='olive', width=0.5)
        ax.set_xlabel("Index of topics")  #设置x轴标签
        ax.set_ylabel("Number of documents")  #设置y轴标签
        ax.set_title("Topic distribution of the test set")  #设置标题
        plt.show()  #显示图像
        
        return p_ts_docs

#外部函数，测试模型及参数K的选择时调用
def plt_perplexity(topics, perplextiy):
    x = topics
    y = perplextiy
    plt.plot(x, y, color="red", linewidth=2)
    plt.xlabel("Number of Topics")
    plt.ylabel("Perplexity")
    print("Perplexity curve under different number of topics is...")
    plt.show()
    
def plt_kl(topics, klList):
    x = topics
    y = klList
    plt.plot(x, y, color="blue", linewidth=2)
    plt.xlabel("Number of Topics")
    plt.ylabel("Kullback-Leibler divergence")
    print("K-L divergence curve under different number of topics is...")
    plt.show()