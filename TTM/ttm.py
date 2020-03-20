# -*- coding: utf-8 -*-
'''
@author: josephqian
@contact: jianpeiqian1990@gmail.com
@software: jupyter
@file: ttm.py
@initialtime:  16/2/20 00:05 PM
@completetime: 26/2/20 04:00 PM
'''
# log
'''
@version: v1.10-202003072130
@modification:
    __init__(three parameters,dictionary);
    likelihood(data);
    inf_gibbs_sampling增加参数phi;
    likelihood增加参数theta和phi;
    plt_likelihood()区分训练与测试
@version: v1.20-202003082000
@modification: 
    docs_time的格式为:day1-day2-……-groupid-drivedate-destination, N需减3;
    inf_init和gibbs_sampling_for_t采样多次t，根据成员人数确定——后证明此策略无效，会让k-t分布趋近于一种t主导;
    time的推断采取先取最高概率topic，再取对应topic最高概率time的策略;
    将get_doc_topic和inf_doc_topic中保存csv的功能转移到类外，同时增加self.docs_time和self.new_docs_time的出发日期和目的地列;
    inf_estimate中增加参数newtime,模型训练时需要从外部传入time_test;
    采用了新的time预测方式：max_inf_doc_time()
@version: v1.21-202003092000
@modification:
    增加函数top2_inf_doc_time(self)，用于计算HitRatio@T=2
@version: v1.22-202003102000
@modification:
    增加函数inf_doc_topic(self)，time数据中增加members列
@version: v1.23-202003111120
@modification:
    avg_kl计算中，多乘了系数2，删除
@version: v1.31-202003131900
@modification:
    增加函数inf_gibbs_sampling_for_t，使得模型测试过程中依然可以使用t的信息
'''

import random
import numpy as np
import pandas as pd
from gensim import corpora
import datetime
import matplotlib.pyplot as plt
import scipy.stats
from sklearn.metrics import *

class TopicTimeModel:
    #主模型初始化
    def __init__(self, data, dictionary, starttime, coef_alpha, beta, gamma, NumberOfIterations, K ):
        #模型变量定义
        self.alpha = coef_alpha/K
        self.beta = beta
        self.gamma = gamma                                     #addddddddddddddddd
        self.NumberOfIterations = NumberOfIterations
        self.K = K
        #引入word相关变量
        self.docs = data
        self.M = len(self.docs)
        self.dictionary = dictionary
        self.V = len(self.dictionary)
        #引入start_time相关变量
        self.docs_time = starttime                                #addddddddddddddddd
        self.dictionary_time = corpora.Dictionary(self.docs_time[:,:1])       #addddddddddddddddd
        self.L = len(self.dictionary_time)                          #addddddddddddddddd
        #其他变量
        self.likelihoodList = [] * self.NumberOfIterations
        #计数变量定义
        self.z = [[]for _ in range(self.M)]                                             
        self.nw = [[0]*self.V for _ in range(self.K)]
        self.nd = [[0]*self.K for _ in range(self.M)]                  #add——计入t对应的主题k
        self.nwsum = [0]*self.K
        self.ndsum = [0]*self.M                                 #add——计入t对应的主题k
        #引入start_time相关计数变量
        self.z_for_t = [[]for _ in range(self.M)]                    #addddddddddddddddd
        self.nt = [[0]*self.L for _ in range(self.K)]                  #addddddddddddddddd
        self.ntsum = [0]*self.K                                 #addddddddddddddddd       
        #模型计数向量和计数矩阵初始化 
        print("1. Begin to initialize the LDA model...")  
        for m in range(self.M):
            #word相关变量初始化
            N = len(self.docs[m])             
            for n in range(N):      
                self.z[m].append(0)            
            for n in range(N):      
                k = random.randint(0, self.K-1)
                self.z[m][n] = k              
                w = self.docs[m][n]
                wid = self.dictionary.token2id[w]
                self.nw[k][wid] += 1
                self.nwsum[k] += 1
                self.nd[m][k] += 1                
            #start_time相关变量初始化
            N_time = len(self.docs_time[m]) - 3
            for n in range(N_time):
                self.z_for_t[m].append(0)
            for n in range(N_time):
                k_for_t = random.randint(0, self.K-1)                       #addddddddddddddddd
                self.z_for_t[m][n] = k_for_t                             #addddddddddddddddd
                t = self.docs_time[m][n]                                #addddddddddddddddd
                tid = self.dictionary_time.token2id[t]                      #addddddddddddddddd
                self.nt[k_for_t][tid] += 1                               #addddddddddddddddd
                self.ntsum[k_for_t] += 1                                #addddddddddddddddd
                self.nd[m][k_for_t] += 1                                #addddddddddddddddd
            self.ndsum[m] = N + N_time                                   #add——再加1
        print(">>> Topic assignment done!!")
        
    #用于测试数据时调用，进行初始化
    def inf_init(self):
        self.new_z = [[]for _ in range(self.new_M)]
        self.new_z_for_t = [[]for _ in range(self.new_M)]
        self.new_nd = [[0]*self.K for _ in range(self.new_M)]
        self.new_ndsum = [0]*self.new_M
        #对新文本中每个doc及其time进行初始化采样
        for m in range(self.new_M):
            N = len(self.new_docs[m])
            for n in range(N):
                self.new_z[m].append(0)
            for n in range(N):
                k = random.randint(0, self.K-1)
                self.new_z[m][n] = k              
                self.new_nd[m][k] += 1
            self.new_ndsum[m] = N
            
            #start_time相关变量初始化            
            self.new_z_for_t[m].append(0)                        #addddddddddddddddd
            new_k_for_t = random.randint(0, self.K-1)               #addddddddddddddddd
            self.new_z_for_t[m][0] = new_k_for_t                   #addddddddddddddddd
            self.new_nd[m][new_k_for_t] += 1                      #addddddddddddddddd
            self.new_ndsum[m] = N+1                            #add——再加1

    def utils_ComputeTransProb(self, m, n):
        w = self.docs[m][n]
        wid = self.dictionary.token2id[w]
        p = [0] * self.K
        Kalpha = self.K * self.alpha
        Vbeta = self.V * self.beta
        for k in range(self.K):
            p[k] = (float(self.nw[k][wid]) + self.beta) / (float(self.nwsum[k]) + Vbeta) * (float(self.nd[m][k]) + self.alpha)
        return p
    
    def utils_ComputeTransProb_time(self, m, n):         #增加参数n
        t = self.docs_time[m][n]
        tid = self.dictionary_time.token2id[t]
        p = [0] * self.K
        Kalpha = self.K * self.alpha
        Lgamma = self.L * self.gamma
        for k in range(self.K):
            p[k] = (float(self.nt[k][tid]) + self.gamma) / (float(self.ntsum[k]) + Lgamma) * (float(self.nd[m][k]) + self.alpha)
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
                phi[k][w] = (float(self.nw[k][w]) + self.beta) / (float(self.nwsum[k]) + Vbeta)
        return phi
 
    def output_ComputePsi(self):
        psi = [[0]*self.L for _ in range(self.K)]
        Lgamma = self.L * self.gamma
        for k in range(self.K):
            for t in self.dictionary_time:
                psi[k][t] = (float(self.nt[k][t]) + self.gamma) / (float(self.ntsum[k]) + Lgamma)
        return psi      

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

    def gibbs_sampling_for_t(self,m,n):             #增加参数n
        k = self.z_for_t[m][n]
        #step1:对应位置上的时间的ID
        t = self.docs_time[m][n]
        tid = self.dictionary_time.token2id[t]
        self.nt[k][tid] -= 1
        self.nd[m][k] -= 1
        self.ntsum[k] -= 1
        #step2:完全条件概率计算，应用Gibbs采样公式得到每个词的主题选择概率
        p = self.utils_ComputeTransProb_time(m,n)               #增加参数n
        #Gibbs采用公式得到的结果是正比关系非等式关系，因此p需要归一化，
        p_norm=list(map(lambda x:x*(1/sum(p)),p))       
        #step3:多项式分布采样/更新主题
        k_new = self.utils_MultiSample(p_norm)    
        #step4:更新计数
        self.nt[k_new][tid] += 1
        self.nd[m][k_new] += 1
        self.ntsum[k_new] += 1
        return k_new
           
    #测试数据的Gibbs采样算法
    def inf_gibbs_sampling(self, m, n, phi):       #增加参数phi
        #phi =  self.output_ComputePhi()         #调整到循环外计算 
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
    
    #测试数据的Gibbs采样算法
    def inf_gibbs_sampling_for_t(self, m, psi):       #增加参数psi
        #psi =  self.output_ComputePsi()             #调整到循环外计算 
        #获取要更新主题的位置（m，n）上对应的词wid
        k = self.new_z_for_t[m][0]
        t = self.new_docs_time[m][0]
        tid = self.dictionary_time.token2id[t]
        #词wid对应的计数减1
        self.new_nd[m][k] -= 1
        #计算wid的主题分布（完全条件概率公式）
        p = [0]*self.K
        for _k in range(self.K):            
            p[_k] = psi[_k][tid] * (float(self.new_nd[m][_k]) + self.alpha)
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
                #先采样word
                N = len(self.docs[m])
                for n in range(N):
                    k_new = self.gibbs_sampling(m, n)
                    self.z[m][n] = k_new
                #再采样time
                N_times = len(self.docs_time[m]) - 3
                for n in range(N_times):               #增加time中重复元素采样
                    k_for_t_new = self.gibbs_sampling_for_t(m,n)        #adddddddddddddd
                    self.z_for_t[m][n] = k_for_t_new                #adddddddddddddd
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
    def inf_estimate(self, newdata, newtime, NewIterations=None):           #增加参数time
        print("7. Begin to test the LDA model...")
        starttime = datetime.datetime.now()
        if NewIterations == None:
            NumberOfIterations = self.NumberOfIterations
        else:
            NumberOfIterations = NewIterations
        self.new_likelihoodList = [] * NumberOfIterations               #inf循环次数改变时，增加该list，完成likelihood计算与绘制
        self.new_docs = newdata
        self.new_M = len(self.new_docs)
        self.new_docs_time = newtime
        self.inf_init()   
        #to estimate the topic structure of unseen documents (querying)                               
        phi = self.output_ComputePhi()          #增加，先计算phi
        psi = self.output_ComputePsi()          #增加，先计算psi
        for i in range(1, NumberOfIterations + 1):
            for m in range(self.new_M):
                N = len(self.new_docs[m])
                for n in range(N):
                    k_new = self.inf_gibbs_sampling(m, n, phi)       #参数中增加phi
                    self.new_z[m][n] = k_new
                #再采样time
                k_for_t_new = self.inf_gibbs_sampling_for_t(m, psi)    #adddddddddddddd
                self.new_z_for_t[m][0] = k_for_t_new                #adddddddddddddd
            #每对语料库循环一次gibbs采样就计算一次似然
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
        drivedate = self.docs_time[:,-2]
        destination = self.docs_time[:,-1]
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
        p_tr_docs = pd.DataFrame(index=range(self.M),columns=["drivedate","destination","documents","topic","probability"])
        p_tr_docs["drivedate"] = drivedate
        p_tr_docs["destination"] = destination
        p_tr_docs["documents"] = documents
        p_tr_docs["topic"] = topic
        p_tr_docs["probability"] = probability
        #p_tr_docs.to_csv('prediction_of_train_documents.csv',encoding='gbk')
        
        print("6. Topic distribution of the traning set is...")
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

    #调用该函数输出：应用最优K的LDA模型主题-时间分布结果
    def get_topic_time(self, k):
        psi = self.output_ComputePsi()
        timeList = []
        probList = []
        for i,t in enumerate(psi[k]):            
            starttime = dict(self.dictionary_time.items())[i]     
            timeList.append(starttime)
            probList.append(round(t,5))
            
        result = [[] * 2 for _ in range(self.L)]
        for _t in range(self.L):
            result[_t].append(timeList[_t])
            result[_t].append(probList[_t])       
        
        result_sort = sorted(result, key=lambda x: (x[1], x[0]), reverse = True)        
        timeList_sort = [time[0] for time in result_sort]
        probList_sort = [prob[1] for prob in result_sort]
        
        print('>>>第',k,'个主题最频繁的时间与概率依次为：')
        print('>>>词：',timeList_sort)
        print('>>>概率： ',probList_sort)
        
    #调用该函数输出：测试数据集在LDA模型上的文档-主题分布结果
    def inf_doc_topic(self):
        drivedate = self.new_docs_time[:,-2]
        destination = self.new_docs_time[:,-1]
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
        p_ts_docs = pd.DataFrame(index=range(self.new_M),columns=["drivedate","destination","documents","topic","probability"])
        p_ts_docs["drivedate"] = drivedate
        p_ts_docs["destination"] = destination
        p_ts_docs["documents"] = documents
        p_ts_docs["topic"] = topic
        p_ts_docs["probability"] = probability
        #p_ts_docs.to_csv('prediction_of_test_documents.csv',encoding='gbk')
        
        print("8. Topic distribution of the test set is...")
        fig,ax=plt.subplots()
        ax.bar(range(self.K),stata_topics,color='olive', width=0.5)
        ax.set_xlabel("Index of topics")  #设置x轴标签
        ax.set_ylabel("Number of documents")  #设置y轴标签
        ax.set_title("Topic distribution of the test set")  #设置标题
        plt.show()  #显示图像
        
        return p_ts_docs

    #调用该函数输出：测试数据集在LDA模型上的文档-时间预测结果——标准计算方式
    def inf_doc_time(self):              
        new_theta = np.mat(self.inf_ComputeTheta())
        psi = np.mat(self.output_ComputePsi())
        omega = np.dot(new_theta,psi)
        pred_time = []
        for m in range(self.new_M):
            tix = np.argmax(omega[m])
            tname = dict(self.dictionary_time.items())[tix] 
            pred_time.append(tname)
        return pred_time
    
        #调用该函数输出：测试数据集在LDA模型上的文档-时间预测结果——标准计算方式
    def top2_inf_doc_time(self):              
        new_theta = np.array(self.inf_ComputeTheta())
        psi = np.array(self.output_ComputePsi())
        omega = np.dot(new_theta,psi)
        pred_time = [[] for _ in range(self.new_M)]
        for m in range(self.new_M):
            #寻找最大值
            tix_1st = np.argmax(omega[m])
            tname_1st = dict(self.dictionary_time.items())[tix_1st] 
            pred_time[m].append(tname_1st)
            #寻找第二大值
            omega[m][tix_1st] = 0
            tix_2nd = np.argmax(omega[m])
            tname_2nd = dict(self.dictionary_time.items())[tix_2nd] 
            pred_time[m].append(tname_2nd)
        return pred_time
    
    #调用该函数输出：测试数据集在LDA模型上的文档-时间预测结果——连续取两次max；这个结果精度不如标准方式
    def max_inf_doc_time(self): 
        new_theta = np.mat(self.inf_ComputeTheta())
        psi = np.mat(self.output_ComputePsi())
        max_pred_time=[]
        for m in range(self.new_M):
            max_kix = np.argmax(new_theta[m])
            max_tix = np.argmax(psi[max_kix])
            max_tname = dict(self.dictionary_time.items())[max_tix] 
            max_pred_time.append(max_tname)
        return max_pred_time        
        
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