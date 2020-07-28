import torch
import numpy as np
from torch.nn.utils import prune


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

    def fix_state(self,modelpath='model/initstate.pth'):
        cuda_using = next(self.parameters()).is_cuda
        if cuda_using:
            self.cpu()
            torch.save(self.state_dict(), modelpath)
            self.cuda()
        else:
            torch.save(self.state_dict(), modelpath)

    def load_state(self, modelpath='model/initstate.pth'):
        cuda_using = next(self.parameters()).is_cuda
        if cuda_using:
            self = SkipGramModel(self.vocab_size,self.emb_dimension)
            self.load_state_dict(torch.load(modelpath))
            self.cuda()
        else:
            self.load_state_dict(torch.load(modelpath))

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
