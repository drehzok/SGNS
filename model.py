import torch
import numpy as np
from torch.nn.utils import prune
import networkx as nx
import collections
import matplotlib.pyplot as plt


class SkipGramModel(torch.nn.Module):

    def __init__(self, vocab_size, emb_dimension):
        super(SkipGramModel, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = torch.nn.Embedding(vocab_size, emb_dimension)
        self.v_embeddings = torch.nn.Embedding(vocab_size, emb_dimension)

        initrange = 1.0 / self.emb_dimension
        torch.nn.init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        torch.nn.init.constant_(self.v_embeddings.weight.data, 0)

    def prune_step(self, pstep, prune_method = prune.l1_unstructured):
        prune_method(self.u_embeddings, name='weight', amount=pstep)
        prune_method(self.v_embeddings, name='weight', amount=pstep)

    def fix_state(self,modelpath='/raid/zhassylbekov/sungbae/model/initstate.pth'):
        cuda_using = next(self.parameters()).is_cuda
        if cuda_using:
            self.cpu()
            torch.save(self.state_dict(), modelpath)
            self.cuda()
        else:
            torch.save(self.state_dict(), modelpath)

    def load_state(self, modelpath='/raid/zhassylbekov/sungbae/model/initstate.pth'):
        umask = dict(self.u_embeddings.named_buffers())['weight_mask'].cpu()
        vmask = dict(self.v_embeddings.named_buffers())['weight_mask'].cpu()
        cuda_using = next(self.parameters()).is_cuda
        self = SkipGramModel(self.vocab_size,self.emb_dimension)
        if cuda_using:
            self.load_state_dict(torch.load(modelpath))
            prune.custom_from_mask(self.u_embeddings,name='weight',mask=umask)
            prune.custom_from_mask(self.v_embeddings,name='weight',mask=vmask)
            self.cuda()
        else:
            self.load_state_dict(torch.load(modelpath))
            prune.custom_from_mask(self.u_embeddings,name='weight',mask=umask)
            prune.custom_from_mask(self.v_embeddings,name='weight',mask=vmask)

    def forward(self, pos_u, pos_v, neg_v):
        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        emb_neg_v = self.v_embeddings(neg_v)

        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -torch.nn.functional.logsigmoid(score)

        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(torch.nn.functional.logsigmoid(-neg_score), dim=1)

        return torch.mean(score + neg_score)

    def save_embedding(self, id2word, file_name):
        embedding = self.u_embeddings.weight.cpu().data.numpy()
        with open(file_name, 'w') as f:
            f.write('%d %d\n' % (len(id2word), self.emb_dimension))
            for wid, w in id2word.items():
                e = ' '.join(map(lambda x: str(x), embedding[wid]))
                f.write('%s %s\n' % (w, e))

    def graph_clustering(self,fname='1.png'):
        w_emb = self.u_embeddings.weight.cpu().data.numpy()
        c_emb = self.v_embeddings.weight.cpu().data.numpy()
        n = self.vocab_size
        edgepair = []
        for i in range(n):
            for j in range(i+1,n):
                m = np.dot(w_emb[i,:],c_emb[j,:])
                p = 1/(np.exp(-m)+1)
                adj = np.random.binomial(1, p)
                if adj == 1:
                    edgepair.append( (i+1,j+1))
        G = nx.Graph()
        G.add_nodes_from(range(1,n+1))
        G.add_edges_from(edgepair)
        clcoef = nx.average_clustering(G)

        degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
# print "Degree sequence", degree_sequence
        degreeCount = collections.Counter(degree_sequence)
        deg, cnt = zip(*degreeCount.items())

        fig, ax = plt.subplots()
        plt.bar(deg, cnt, width=0.80, color='b')

        plt.title("Degree Histogram, clustering = %.6f" % clcoef)
        plt.ylabel("Count")
        plt.xlabel("Degree")
        ax.set_xticks([d + 0.4 for d in deg])
        ax.set_xticklabels(deg)
        path = '/raid/zhassylbekov/sungbae/figs'
        path += '/' + fname
        plt.savefig(path)


        




'''
class LogitSGNSModel(torch.nn.Module):

    def __init__(self, vocab_size, emb_dimension, epsilon):
        super(LogitSGNSModel, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = torch.nn.Embedding(vocab_size, emb_dimension)
        self.v_embeddings = torch.nn.Embedding(vocab_size, emb_dimension)
        self.eps = epsilon

        initrange = 1.0 / np.sqrt(self.emb_dimension)
        torch.nn.init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        torch.nn.init.uniform_(self.v_embeddings.weight.data, -initrange, initrange)

    def forward(self, pos_u, pos_v, neg_v):
        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        emb_neg_v = self.v_embeddings(neg_v)

        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = torch.clamp(score, min=self.eps, max=1-self.eps)
        score = -torch.log(score)

        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, min=self.eps, max=1-self.eps)
        neg_score = -torch.sum(torch.log(1-neg_score), dim=1)

        return torch.mean(score + neg_score)

    def save_embedding(self, id2word, file_name):
        embedding = self.u_embeddings.weight.cpu().data.numpy()
        with open(file_name, 'w') as f:
            f.write('%d %d\n' % (len(id2word), self.emb_dimension))
            for wid, w in id2word.items():
                e = ' '.join(map(lambda x: str(x), embedding[wid]))
                f.write('%s %s\n' % (w, e))
'''
