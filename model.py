import torch
import numpy as np
from torch.nn.utils import prune
import networkx as nx
import networkit as nk
import collections
import matplotlib.pyplot as plt
import scipy


class SkipGramModel(torch.nn.Module):

    def __init__(self, vocab_size, emb_dimension):
        super(SkipGramModel, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = torch.nn.Embedding(self.vocab_size, self.emb_dimension)
        self.v_embeddings = torch.nn.Embedding(self.vocab_size, self.emb_dimension)

        initrange = 1.0 / self.emb_dimension
        torch.nn.init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        torch.nn.init.constant_(self.v_embeddings.weight.data, 0)

    def prune_step(self, pstep, prune_mode = 'classic'):
        if prune_mode == 'classic':
            prune.l1_unstructured(self.u_embeddings, name='weight', amount=pstep)
            prune.l1_unstructured(self.v_embeddings, name='weight', amount=pstep)
        else:
            #non-l1 simple pruning scheme
            self.prune_step_change(pstep,prune_mode)


    def prune_step_change(self, pstep, prune_mode):
        cuda_using = next(self.parameters()).is_cuda
        #reimport fixed u and v weights
        ui,vi = self.load_weights()
        #fix current weights
        uc, vc = ( self.u_embeddings.weight.data.clone().cpu(),
                self.v_embeddings.weight.data.clone().cpu() )
        #fix current masks
        if not list(self.u_embeddings.named_buffers()):
            prune.Identity(self.u_embeddings, name='weight')
            prune.Identity(self.v_embeddings, name='weight')
        umask = dict(self.u_embeddings.named_buffers())['weight_mask'].cpu()
        vmask = dict(self.v_embeddings.named_buffers())['weight_mask'].cpu()

        u_temp = torch.nn.Embedding(self.vocab_size, self.emb_dimension)
        v_temp = torch.nn.Embedding(self.vocab_size, self.emb_dimension)

        if prune_mode=='change':
            f = lambda x,y: x-y
        elif prune_mode=='absolute change':
            f = lambda x,y: torch.abs(x) - torch.abs(y)
        else:
            f = lambda x,y: x
        # weights to be left must have higher function outputs
        u_temp.weight.data.copy_(f(uc,ui))
        v_temp.weight.data.copy_(f(vc,vi))


        prune.custom_from_mask(u_temp,name='weight',mask=umask)
        prune.custom_from_mask(v_temp,name='weight',mask=vmask)
        if cuda_using:
            u_temp.cuda()
            v_temp.cuda()
        prune.l1_unstructured(u_temp, name='weight', amount=pstep)
        prune.l1_unstructured(v_temp, name='weight', amount=pstep)
        #checked, cuda <-> cpu crash DNE
        u_temp.weight.data.copy_(uc)
        v_temp.weight.data.copy_(vc)

        self.u_embeddings = u_temp
        self.v_embeddings = v_temp



    def fix_state(self, targetpath='/raid/zhassylbekov/sungbae/model/'):
        cuda_using = next(self.parameters()).is_cuda
        if cuda_using:
            self.cpu()
            torch.save(self.state_dict(), targetpath+'initstate.pth')
            self.cuda()
        else:
            torch.save(self.state_dict(), targetpath+'initstate.pth')

    def load_weights(self,targetpath='/raid/zhassylbekov/sungbae/model/'):
        temp = SkipGramModel(self.vocab_size, self.emb_dimension)
        temp.load_state_dict(torch.load(targetpath+'initstate.pth'))
        return temp.u_embeddings.weight.data.clone(), temp.v_embeddings.weight.data.clone()

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
        Adjmat = np.zeros((n,n), dtype=bool)
        for i in range(n):
            for j in range(i+1,n):
                m = np.dot(w_emb[i,:],c_emb[j,:])
                p = 1/(np.exp(-m)+1)
                adj = np.random.binomial(1, p)
                if adj == 1:
                    Adjmat[i,j] = True
                    Adjmat[j,i] = True

        Adjmat = scipy.sparse.csr_matrix(Adjmat)
        G = nx.from_scipy_sparse_matrix(Adjmat.tocsr())

        G_nk = nk.nxadapter.nx2nk(G)

        clcoef = nk.globals.clustering(G_nk,error=0.0005)

        fig, ax = plt.subplots()
        degrees = [G.degree(n) for n in G.nodes()]
        ax.hist(degrees,bins=80,range=(0,20000))

        plt.title("Degree Histogram, clustering = %.6f" % clcoef)
        plt.ylabel("Count")
        plt.xlabel("Degree")


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
