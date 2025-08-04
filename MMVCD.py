import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import os, random
import json, nltk
import torch
from torch import nn
from torch.cuda import amp
from transformers import BertTokenizer, get_linear_schedule_with_warmup
import torch.nn.functional as F
from torch.autograd import Function
from transformers import BertModel, BertConfig
from nltk import word_tokenize     #以空格形式实现分词
from nltk.corpus import stopwords
import transformers
from torch.optim import Optimizer
from torch.distributions.bernoulli import Bernoulli
import math
from sklearn import svm

model_path = "../input/huggingface-bert/bert-base-uncased/"  #BertModel BertTokenizer BertConfig
tokenizer = transformers.BertTokenizer.from_pretrained(model_path, do_lower_case=True)
model_config = transformers.BertConfig.from_pretrained(model_path)

model_config.output_hidden_states = True

MAX_LENTH = 256

BATCH = 16


class ForeverDataIterator:
    r"""A data iterator that will never stop producing data"""
    def __init__(self, data_loader: DataLoader, device=None):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)
        self.device = device

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
 
def load_file_all(filename, num_start, num_end):
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines = [line.rstrip('\n') for line in lines]
    return lines[num_start:num_end]

def compute_kl_loss(p, q, pad_mask = None):

    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss

class MMDLoss(nn.Module):
    '''
    计算源域数据和目标域数据的MMD距离
    Params:
    source: 源域数据（n * len(x))
    target: 目标域数据（m * len(y))
    kernel_mul:
    kernel_num: 取不同高斯核的数量
    fix_sigma: 不同高斯核的sigma值
    Return:
    loss: MMD loss
    '''
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target, s_label=None, t_label=None):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss
        elif self.kernel_type == 'cmmd':
            batch = s_label.size()[0]
            s_label = s_label.cpu()

            s_label = s_label.view(batch,1)
            s_label = torch.zeros(batch, batch-1).scatter_(1, s_label.data, 1)
            s_label = Variable(s_label).cuda()

            t_label = t_label.cpu()
            t_label = t_label.view(batch, 1)
            t_label = torch.zeros(batch, batch-1).scatter_(1, t_label.data, 1)
            t_label = Variable(t_label).cuda()

            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(source, target,
                                      kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            loss = torch.mean(torch.mm(s_label, torch.transpose(s_label, 0, 1)) * XX +
                              torch.mm(t_label, torch.transpose(t_label, 0, 1)) * YY -
                              2 * torch.mm(s_label, torch.transpose(t_label, 0, 1)) * XY)
            return loss

def proxy_a_distance(source_X, target_X, verbose=False):
    """
    Compute the Proxy-A-Distance of a source/target representation
    """
    nb_source = np.shape(source_X)[0]
    nb_target = np.shape(target_X)[0]

    if verbose:
        print('PAD on', (nb_source, nb_target), 'examples')

    C_list = np.logspace(-5, 4, 10)

    half_source, half_target = int(nb_source/2), int(nb_target/2)
    train_X = np.vstack((source_X[0:half_source, :], target_X[0:half_target, :]))
    train_Y = np.hstack((np.zeros(half_source, dtype=int), np.ones(half_target, dtype=int)))

    test_X = np.vstack((source_X[half_source:, :], target_X[half_target:, :]))
    test_Y = np.hstack((np.zeros(nb_source - half_source, dtype=int), np.ones(nb_target - half_target, dtype=int)))

    best_risk = 1.0
    for C in C_list:
        clf = svm.SVC(C=C, kernel='linear', verbose=False)
        
        clf.fit(train_X, train_Y)
 
        train_risk = np.mean(clf.predict(train_X) != train_Y)
        test_risk = np.mean(clf.predict(test_X) != test_Y)

        if verbose:
            print('[ PAD C = %f ] train risk: %f  test risk: %f' % (C, train_risk, test_risk))

        if test_risk > .5:
            test_risk = 1. - test_risk

        best_risk = min(best_risk, test_risk)

    return 2 * (1. - 2 * best_risk)

def estimate_mu(_X1, _Y1, _X2, _Y2):
    """
    Estimate value of mu using conditional and marginal A-distance.
    """
    adist_m = proxy_a_distance(_X1, _X2)

    Cs, Ct = np.unique(_Y1), np.unique(_Y2)
    C = np.intersect1d(Cs, Ct)
    epsilon = 1e-3
    list_adist_c = []
    tc = len(C)
    for i in C:
        ind_i, ind_j = np.where(_Y1 == i), np.where(_Y2 == i)
        Xsi = _X1[ind_i[0], :]
        Xtj = _X2[ind_j[0], :]
        if len(Xsi) <= 1 or len(Xtj) <= 1:
            tc -= 1
            continue
        adist_i = proxy_a_distance(Xsi, Xtj)
        list_adist_c.append(adist_i)
    if tc < 1:
        return 0
    adist_c = sum(list_adist_c) / tc
    mu = adist_c / (adist_c + adist_m)
    if mu > 1:
        mu = 1
    if mu < epsilon:
        mu = 0
    return mu

    
class cnn_extractor(nn.Module):
    def __init__(self, feature_kernel, input_size):
        super(cnn_extractor, self).__init__()
        self.convs = torch.nn.ModuleList(
            [torch.nn.Conv1d(input_size, feature_num, kernel)
             for kernel, feature_num in feature_kernel.items()])
        input_shape = sum([feature_kernel[kernel] for kernel in feature_kernel])

    def forward(self, input_data):
        share_input_data = input_data.permute(0, 2, 1)
        feature = [F.relu(conv(share_input_data)) for conv in self.convs]
        feature = [torch.max_pool1d(f, f.shape[-1]) for f in feature]
        feature = F.dropout(torch.cat(feature, dim=1), p=0.1)
        feature = feature.view([-1, feature.shape[1]])
        return feature
    
class MLP(torch.nn.Module):

    def __init__(self, input_dim, embed_dim, dropout):
        super().__init__()
        layers = list()
        layers.append(torch.nn.Linear(input_dim, embed_dim))
        #layers.append(torch.nn.BatchNorm1d(embed_dim))
        layers.append(torch.nn.LayerNorm(embed_dim))
        layers.append(torch.nn.ReLU(True))
        layers.append(torch.nn.Dropout(p=dropout))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)
    
class MaskAttention(torch.nn.Module):
    """
    Compute attention layer
    """
    def __init__(self, input_shape):
        super(MaskAttention, self).__init__()
        self.attention_layer = torch.nn.Linear(input_shape, 1)

    def forward(self, inputs, mask=None):
        # print("inputs: ", inputs.shape)     #(128, 170, 768)
        scores = self.attention_layer(inputs).view(-1, inputs.size(1))
        # print("scores: ", scores.shape)     #(128, 170)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        scores = torch.softmax(scores, dim=-1).unsqueeze(1)
        # print("scores: ", scores.shape)     #(128, 1, 170)
        outputs = torch.matmul(scores, inputs).squeeze(1)
        # print("outputs: ", outputs.shape)   #(128, 768)

        return outputs, scores
    
class Our_Token(nn.Module):
    def __init__(self, trainable=False):
        super(Our_Token, self).__init__()
        self.config = model_config
        self.bert = transformers.BertModel.from_pretrained(model_path, config=model_config).requires_grad_(False)
        
        #self.bert.embeddings.requires_grad_(False)
        self.bert.encoder.layer[-4:].requires_grad_(True)
        #self.bert.pooler.requires_grad_(False)
        
        self.extractor = MLP(768, 384, 0.1)


        self.classifier = nn.Linear(384, 2)

        
        self.attention = MaskAttention(768)
        
    def forward(self, x_origin, mask_origin):
        bert_out = self.bert(input_ids=x_origin, attention_mask=mask_origin)
        
        last_hidden = bert_out.last_hidden_state
        
        gate_input_feature, _ = self.attention(last_hidden[:,1:,:], mask_origin[:,1:])

        token_f = self.extractor(gate_input_feature)
       
        result = self.classifier(token_f)
        
        out_data = {'logits': result, 'feature': token_f}

        return out_data


def ProcessData(X, Y):
    Inputid = []
    Attenmask = []
    Labels = []
    
    for data, label in zip(X, Y):
        text = data
        
        encode_dict = tokenizer.encode_plus(text, max_length=MAX_LENTH, padding="max_length", truncation=True, add_special_tokens=True)
        input_ids = encode_dict["input_ids"]
        atten_mask = encode_dict["attention_mask"]
        
#         k = 0
#         for in_id, at_mask in zip(input_ids, atten_mask):
#             if in_id == 103:#103
#                 atten_mask[k] = 0
#             k = k + 1

        Labels.append(label)
        Inputid.append(input_ids)
        Attenmask.append(atten_mask)

    Inputid = torch.tensor(Inputid, dtype=torch.long).to(device)
    Attenmask = torch.tensor(Attenmask, dtype=torch.long).to(device)
    Labels = torch.tensor(Labels, dtype=torch.long).to(device)
    

    return Inputid, Attenmask, Labels


def test_model(model, test_dataloader):

    avg_acc = []
    model.to(device)
    model.eval()

    test_pre = []
    test_label = []
    test_result = []
    correct = 0
    with torch.no_grad():
        for batch in test_dataloader:
            
            target_input_origin, target_mask_origin, target_labels = batch

            test_label.extend(target_labels.cpu())

            t_result = model(target_input_origin, target_mask_origin)
            #t_result = t_result[0:len(t_result)//2]
            temp = t_result['logits'].cpu().numpy()
            test_result.extend(temp[:,1])
            
            test_pre.extend(torch.max(t_result['logits'], 1)[1].cpu())
            
        
        acc = sum(p == t for p, t in zip(test_pre, test_label)) / len(test_pre)

    return acc, test_pre, test_label, test_result

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_cpu = torch.device("cpu")

def train_model(C_model, train_dataset_source, train_dataset_target, test_dataset, mylr, my_epoch=10):
    EPOCH = my_epoch
 
    class_steps = len(train_dataset_source) * EPOCH
    

    C_optimizer = torch.optim.Adam(C_model.parameters(), lr=mylr)
    C_schedule = get_linear_schedule_with_warmup(C_optimizer, num_warmup_steps=int(class_steps*0.1), num_training_steps=class_steps)

    MMD = MMDLoss()
    CMMD = MMDLoss(kernel_type='cmmd')
   
    ce_criterion = torch.nn.CrossEntropyLoss()
    accum_iter = 1  

    
    for epoch in range(EPOCH):

        C_model.train()
        C_model.to(device)

        train_mmd_loss_batch, train_ce_loss_batch, train_acc_batch, train_acc_batch1 = [], [], [], []
        train_dist_loss_batch, train_cmmd_loss_batch, train_kl_loss_batch, train_con_loss_batch = [], [], [], []
        train_uncon_loss_batch = []
        
        label_, predict_ = [], []
     
        train_source_iter = ForeverDataIterator(train_dataset_source)
        train_target_iter = ForeverDataIterator(train_dataset_target)

        for idx in range(max(len(train_dataset_source), len(train_dataset_target))):
            
            src_data = next(train_source_iter)
            tgt_data = next(train_target_iter)

            src_ids, src_mask, src_ids2, src_mask2, src_ids3, src_mask3, src_label = src_data
            tgt_ids, tgt_mask, tgt_ids2, tgt_mask2, tgt_ids3, tgt_mask3, tgt_label = tgt_data
            
            all_src_ids = torch.cat([src_ids, src_ids2, src_ids3], 0)
            all_src_mask = torch.cat([src_mask, src_mask2, src_mask3], 0)

            all_tgt_ids = torch.cat([tgt_ids, tgt_ids2, tgt_ids3], 0)
            all_tgt_mask = torch.cat([tgt_mask, tgt_mask2, tgt_mask3], 0)
            
            label_.extend(src_label)
            
            C_optimizer.zero_grad()
                     
            
            all_src_output = C_model(all_src_ids, all_src_mask)
            src_result, src_result2, src_result3 = torch.chunk(all_src_output['logits'], 3, 0)
            src_f, src_f2, src_f3 = torch.chunk(all_src_output['feature'], 3, 0)
             
            all_tgt_output = C_model(all_tgt_ids, all_tgt_mask)
            tgt_result, tgt_result2, tgt_result3 = torch.chunk(all_tgt_output['logits'], 3, 0)
            tgt_f, tgt_f2, tgt_f3 = torch.chunk(all_tgt_output['feature'], 3, 0)
            
            source_loss = ce_criterion(src_result, src_label.long()) + ce_criterion(src_result2, src_label.long()) + ce_criterion(src_result3, src_label.long())
            

            tpred2 = F.softmax(tgt_result2, dim=1)

            
            tpred3 = F.softmax(tgt_result3, dim=1)

            kl_loss = compute_kl_loss(src_result, src_result2) + compute_kl_loss(tgt_result, tgt_result2) + compute_kl_loss(src_result, src_result3) + compute_kl_loss(tgt_result, tgt_result3)
            
            
            tpred1 = F.softmax(tgt_result, dim=1)
            # 去除cmmd
            cmmd_loss = CMMD(src_f, tgt_f, src_label.long(), torch.max(tpred1, 1)[1]) 
            
            #去除mmd
            mmd_loss = MMD(src_f, tgt_f)  
            mu = estimate_mu(src_f.detach().cpu().numpy(), src_label.detach().cpu().numpy(), tgt_f.detach().cpu().numpy(), torch.max(tpred1, 1)[1].detach().cpu().numpy())
            if np.isnan(mu):
                mu = 0.5
            #去除mmd end

            all_src_f = torch.cat((src_f2, src_f3), 0)
            all_tgt_f = torch.cat((tgt_f2, tgt_f3), 0)
            all_label = torch.cat((src_label, src_label), 0)
            all_pred = torch.cat((tpred2, tpred3), 0)

            # 去除cmmd
            cmmd_loss23 = CMMD(all_src_f, all_tgt_f, all_label.long(), torch.max(all_pred, 1)[1])
            #去除mmd
            mmd_loss23 = MMD(all_src_f, all_tgt_f)
            mu23 = estimate_mu(all_src_f.detach().cpu().numpy(), all_label.detach().cpu().numpy(), all_tgt_f.detach().cpu().numpy(), torch.max(all_pred, 1)[1].detach().cpu().numpy())
            if np.isnan(mu23):
                mu23 = 0.5
            #去除mmd end

            loss =  0.2*kl_loss + source_loss + cmmd_loss*(1-mu) + mmd_loss * mu + cmmd_loss23*(1-mu23) + mmd_loss23 * mu23

            loss.backward()
            C_optimizer.step()
            C_schedule.step()
            
            predict_.extend(torch.max(src_result, 1)[1])
            
            #_, predict = torch.max(src_output['logits'], 1)
        
        train_acc_batch = sum(p == t for p, t in zip(predict_, label_)) / len(predict_)
        
        C_model.eval()
        label_, predict_ = [], []
        with torch.no_grad():
            for tgt_ids, tgt_mask, tgt_label in test_dataset:
                label_.extend(tgt_label)
                out_test = C_model(tgt_ids, tgt_mask)

                predict_.extend(torch.max(out_test['logits'], 1)[1])
                
            train_acc_batch1 = sum(p == t for p, t in zip(predict_, label_)) / len(predict_)

        print(train_acc_batch, train_acc_batch1)

    return C_model
           
def get_jiao(source_texts, target_texts, stop_words):
    src_all_words = []
    for text in source_texts:
        words = word_tokenize(text.lower())
        filtered_corpus = [w for w in words if not w in stop_words]
        src_all_words.extend(filtered_corpus)
    tgt_all_words = []
    for text in target_texts:
        words = word_tokenize(text.lower())
        filtered_corpus = [w for w in words if not w in stop_words]
        tgt_all_words.extend(filtered_corpus)

    src_all_words_set = set(src_all_words)
    tgt_all_words_set = set(tgt_all_words)

    jiao = src_all_words_set.intersection(tgt_all_words_set)
    src_cha = src_all_words_set.difference(tgt_all_words_set)
    tgt_cha = tgt_all_words_set.difference(src_all_words_set)
    return jiao, src_cha, tgt_cha, src_all_words_set, tgt_all_words_set

def get_mask_data(texts, set_cha, prob=0.3):
    new_texts = []
    for text in texts:
        new_text = []
        words = word_tokenize(text)
        for w in words:
            if w.lower() in set_cha and random.random() <= prob: #
                new_text.append('[MASK]')
            else:
                new_text.append(w)
        new_texts.append(' '.join(new_text))
    return new_texts


def preprocess(corpus):
    """对语料进行分词并统计词频"""
    word_counts = Counter()
    for doc in corpus:
        # 简单分词：按空格拆分
        words = word_tokenize(doc.lower())
        filtered_words = [w for w in words if not w in stop_words]
        # 统计文档词频
        word_counts.update(set(filtered_words))  # 按文档计数，只记录每个文档中词出现一次
    return word_counts
    
def get_sort(class_a_corpus, class_b_corpus, jiao):
    # 计算文档数目
    num_a = len(class_a_corpus)
    num_b = len(class_b_corpus)
    
    # 统计词频
    class_a_word_counts = preprocess(class_a_corpus)
    class_b_word_counts = preprocess(class_b_corpus)
    
    # 找到公共词表
    common_vocab = set(class_a_word_counts.keys()) & set(class_b_word_counts.keys())
    
    # 计算公共词在每类语料中的频率
    num_a_frequencies = {word: class_a_word_counts[word] / num_a for word in common_vocab}
    num_b_frequencies = {word: class_b_word_counts[word] / num_b for word in common_vocab}
    
    # 计算平均值
    average_frequencies = {word: (num_a_frequencies[word] + num_b_frequencies[word]) / 2 for word in common_vocab}
    
    # 按平均值排序
    sorted_vocab = sorted(common_vocab, key=lambda word: average_frequencies[word], reverse=True)
    sorted_avg_freq = [average_frequencies[word] for word in sorted_vocab]
    
    # 输出结果
    sorted_common_vocab = sorted_vocab  # 排序后的公共词表
    sorted_average_frequencies = sorted_avg_freq  # 排序后的平均计数

    return sorted_common_vocab, sorted_average_frequencies
    
    # 打印或保存结果
    # print("类别A排序后的公共词表：", v_a)
    # print("类别A排序后的计数：", sorted_num_a)
    # print("类别B排序后的公共词表：", v_b)
    # print("类别B排序后的计数：", sorted_num_b)
#新加入end


from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

 
def get_dataset(source_texts_share, target_texts_share, source_texts_spec, target_texts_spec, source_texts_train, source_labels_train, target_texts_train, target_labels_train, target_texts_test, target_labels_test, droplast=True):
    tgt_Inputid_train, tgt_Attenmask_train, tgt_Labels_train = ProcessData(target_texts_train, target_labels_train)
    tgt_Inputid_train2, tgt_Attenmask_train2, tgt_Labels_train2 = ProcessData(target_texts_share, target_labels_train)
    tgt_Inputid_train3, tgt_Attenmask_train3, tgt_Labels_train3 = ProcessData(target_texts_spec, target_labels_train)
    tgt_dataset_train = TensorDataset(tgt_Inputid_train, tgt_Attenmask_train, tgt_Inputid_train2, tgt_Attenmask_train2, tgt_Inputid_train3, tgt_Attenmask_train3, tgt_Labels_train)
    tgt_dataloader_train = DataLoader(tgt_dataset_train, shuffle=True, batch_size=BATCH, drop_last=droplast)  #

    src_Inputid_train, src_Attenmask_train, src_Labels_train = ProcessData(source_texts_train, source_labels_train)
    src_Inputid_train2, src_Attenmask_train2, src_Labels_train2 = ProcessData(source_texts_share, source_labels_train)
    src_Inputid_train3, src_Attenmask_train3, src_Labels_train3 = ProcessData(source_texts_spec, source_labels_train)
    src_dataset_train = TensorDataset(src_Inputid_train, src_Attenmask_train, src_Inputid_train2, src_Attenmask_train2, src_Inputid_train3, src_Attenmask_train3, src_Labels_train)
    src_dataloader_train = DataLoader(src_dataset_train, shuffle=True, batch_size=BATCH, drop_last=droplast)  #

    tgt_Inputid_test, tgt_Attenmask_test, tgt_Labels_test = ProcessData(target_texts_test, target_labels_test)
    tgt_dataset_test = TensorDataset(tgt_Inputid_test, tgt_Attenmask_test, tgt_Labels_test)
    tgt_dataloader_test = DataLoader(tgt_dataset_test, shuffle=False, batch_size=1, drop_last=False)  #
    
    return src_dataloader_train, tgt_dataloader_train, tgt_dataloader_test
        
import gc
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn import metrics
import time
import json
import pandas as pd
from collections import Counter

srcnames = ['bestFriend', 'bestFriend',  'deathPenalty',  'deathPenalty', 'abortion', 'abortion', 'restaurant','doctor', 'hotel', 'hotel', 'restaurant', 'doctor']#[::-1]
tgtnames = ['deathPenalty', 'abortion',  'abortion', 'bestFriend',  'deathPenalty', 'bestFriend', 'hotel', 'hotel', 'restaurant', 'doctor', 'doctor', 'restaurant']#[::-1]

 
srcnames = srcnames[9:]
tgtnames = tgtnames[9:]

for src_name, tgt_name in zip(srcnames, tgtnames):
    
    Acc = []
    Auc = []
    Precision = []
    Recall = []
    F1 = []

    Precision_true = []
    Recall_true = []
    F1_true = []

    F1_weight = []
    Recall_weight = []
    Precision_weight = []

    Precision_macro = []
    Recall_macro = []
    F1_macro = []

    Precision_micro = []
    Recall_micro = []
    F1_micro = []

    accs = []
    
    FPR, TPR, AUC = [], [], []
    TIME_trian, TIME_test = [], []
    
    src_len = 200
    if src_name == 'hotel':
        src_len = 800
    source_texts_T = load_file_all('../input/alldomain/' + src_name + '-truth.txt', 0, src_len)
    source_texts_F = load_file_all('../input/alldomain/' + src_name + '-false.txt', 0, src_len)
    source_texts = source_texts_T + source_texts_F
    source_pos_labels = [0 for i in range(len(source_texts_T))]  # (5000, )
    source_neg_labels = [1 for i in range(len(source_texts_F))]   # (5000, )
    source_labels = source_pos_labels + source_neg_labels
 
    tgt_len = 200
    if tgt_name == 'hotel':
        tgt_len = 800
    target_texts_T = load_file_all(r'../input/alldomain/' + tgt_name + '-truth.txt', 0, tgt_len)
    target_texts_F = load_file_all(r'../input/alldomain/' + tgt_name + '-false.txt', 0, tgt_len)
    target_texts = target_texts_T + target_texts_F
    target_pos_labels = [0 for i in range(len(target_texts_T))]  # (5000, )
    target_neg_labels = [1 for i in range(len(target_texts_F))]   # (5000, )
    target_labels = target_pos_labels + target_neg_labels

     
    
    interpunctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
    stop_words = stopwords.words('english') + interpunctuations
    src_tgt_jiao, src_cha, tgt_cha, src_word_set, tgt_word_set = get_jiao(source_texts, target_texts, stop_words)
    
    change_target_texts = target_texts

    #改了大数据集
    for seed in range(0, 5):
        
        set_seed(seed)
        
        a = src_name + " -> " + tgt_name
        print(a)
        CNN_model_source = Our_Token()

         
        src_tgt_jiao, src_cha, tgt_cha, src_word_set, tgt_word_set = get_jiao(source_texts, change_target_texts, stop_words)
        

        
        source_texts_share = get_mask_data(source_texts, src_tgt_jiao, prob=0.3)
        target_texts_share = get_mask_data(target_texts, src_tgt_jiao, prob=0.3)
        
        source_texts_spec = get_mask_data(source_texts, src_tgt_jiao, prob=0.3)
        target_texts_spec = get_mask_data(target_texts, src_tgt_jiao, prob=0.3)
        
        print(len(src_cha), len(tgt_cha))
        print(len(src_tgt_jiao), len(src_word_set)-len(src_tgt_jiao), len(tgt_word_set)-len(src_tgt_jiao))
        
        src_X_train = source_texts# + new_source_texts
        src_Y_train = source_labels# + source_labels
        tgt_X_train = target_texts# + new_target_texts
        tgt_Y_train = target_labels# + target_labels
        tgt_X_test = target_texts#tgt_aug_texts
        tgt_Y_test = target_labels#tgt_aug_labels
        
        
        
        src_dataloader_train, tgt_dataloader_train, tgt_dataloader_test = get_dataset(source_texts_share, target_texts_share, source_texts_spec, target_texts_spec, src_X_train, src_Y_train, tgt_X_train, tgt_Y_train, tgt_X_test, tgt_Y_test)
        print(len(src_X_train), len(tgt_X_train), len(tgt_X_test))
        
        start_train_time = time.time()  # 记录开始时间
        CNN_model_source = train_model(CNN_model_source, src_dataloader_train, tgt_dataloader_train, tgt_dataloader_test, 1e-4, 10) 
        end_train_time = time.time()  # 记录结束时间
        elapsed_train_time = end_train_time - start_train_time  # 计算运行时间

        start_test_time = time.time()  # 记录开始时间
        acc, test_pre, test_lable, test_result = test_model(CNN_model_source, tgt_dataloader_test)
        end_test_time = time.time()  # 记录结束时间
        elapsed_test_time = end_test_time - start_test_time  # 计算运行时间
        
        
        
        del CNN_model_source

        print(a, "当前测试： ", acc)


        val_f1 = f1_score(test_lable, test_pre)
        val_recall = recall_score(test_lable, test_pre)
        val_precision = precision_score(test_lable, test_pre)
        val_acc = accuracy_score(test_lable, test_pre)

        val_f1_true = f1_score(test_lable, test_pre, pos_label=0)
        val_recall_true = recall_score(test_lable, test_pre, pos_label=0)
        val_precision_true = precision_score(test_lable, test_pre, pos_label=0)

        val_f1_macro = f1_score(test_lable, test_pre, average='macro')
        val_recall_macro = recall_score(test_lable, test_pre, average='macro')
        val_precision_macro = precision_score(test_lable, test_pre, average='macro')

        val_f1_micro = f1_score(test_lable, test_pre, average='micro')
        val_recall_micro = recall_score(test_lable, test_pre, average='micro')
        val_precision_micro = precision_score(test_lable, test_pre, average='micro')

        val_f1_weight = f1_score(test_lable, test_pre, average='weighted')
        val_recall_weight = recall_score(test_lable, test_pre, average='weighted')
        val_precision_weight = precision_score(test_lable, test_pre, average='weighted')

        fpr, tpr, thresholds = metrics.roc_curve(test_lable, test_result, pos_label=1)
        auc_value = metrics.auc(fpr, tpr)

        FPR.append(fpr.tolist())
        TPR.append(tpr.tolist())
        AUC.append(auc_value.tolist())
        
        TIME_trian.append(elapsed_train_time)
        TIME_test.append(elapsed_test_time)

        Acc.append(val_acc)
        Precision.append(val_precision)
        Recall.append(val_recall)
        F1.append(val_f1)
        Precision_true.append(val_precision_true)
        Recall_true.append(val_recall_true)
        F1_true.append(val_f1_true)

        Auc.append(auc_value)
        F1_weight.append(val_f1_weight)
        Recall_weight.append(val_recall_weight)
        Precision_weight.append(val_precision_weight)
        Precision_macro.append(val_precision_macro)
        Recall_macro.append(val_recall_macro)
        F1_macro.append(val_f1_macro)
        Precision_micro.append(val_precision_micro)
        Recall_micro.append(val_recall_micro)
        F1_micro.append(val_f1_micro)


        accs.append(acc)

        
        gc.collect()
        print(a, "Our_Token测试： ",seed,  " acc = ", accs)

    print(accs)
    print(np.array(accs).mean())

    print("false:, ", np.mean(Acc), np.mean(Precision), np.mean(Recall), np.mean(F1))
    print("true:, ", np.mean(Acc), np.mean(Precision_true), np.mean(Recall_true), np.mean(F1_true))
    print("macro: ", np.mean(Acc), np.mean(Precision_macro), np.mean(Recall_macro), np.mean(F1_macro))
    print("micro: ", np.mean(Acc), np.mean(Precision_micro), np.mean(Recall_micro), np.mean(F1_micro))
    print("weighted: ", np.mean(Acc), np.mean(Precision_weight), np.mean(Recall_weight), np.mean(F1_weight))
    print("auc: ", np.mean(Auc))
    print(a)
    data_dict = {
        "FPR": FPR,  # 直接获取列表
        "TPR": TPR,
        "AUC": AUC,
        "TIME_trian": TIME_trian,
        "TIME_test": TIME_test,
        "Acc": Acc,
        "Precision": Precision,
        "Recall": Recall,
        "F1": F1,
        "Precision_true": Precision_true,
        "Recall_true": Recall_true,
        "F1_true": F1_true,
        "Precision_macro": Precision_macro,
        "Recall_macro": Recall_macro,
        "F1_macro": F1_macro,
        "Precision_micro": Precision_micro,
        "Recall_micro": Recall_micro,
        "F1_micro": F1_micro,
        "Precision_weight": Precision_weight,
        "Recall_weight": Recall_weight,
        "F1_weight": F1_weight
    }
    
    # 转换为 Pandas DataFrame（可选）
    df = pd.DataFrame([data_dict])  # 以字典创建 DataFrame
    
    # 将数据转换为 JSON 格式
    file_name = "MMVCD-Result" + src_name + "-" + tgt_name
    json_data = df.to_json(file_name, orient="records")