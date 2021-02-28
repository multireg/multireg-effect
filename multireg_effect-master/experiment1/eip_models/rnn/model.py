import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import random
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence

class  YUAN(nn.Module):
    def __init__(self,embedding_matrix,args):#hidden_dim,out_channel,dense_out,n_gram_num,dp):
        super(YUAN, self).__init__()
        num_embeddings,embedding_dim = embedding_matrix.shape
        self.max_len = args.max_len
        self.embedding = nn.Embedding(num_embeddings,embedding_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix,dtype=torch.float))
        #self.embedding.weight.requires_grad = False
        self.lstm1 = nn.LSTM(embedding_dim, args.hs1,
                            num_layers=args.layer_num, bidirectional=args.bidirectional)
        
        self.lstm2 = nn.LSTM(2*args.hs1, args.hs2,
                            num_layers=args.layer_num, bidirectional=args.bidirectional)
        
        # Dense Layer weights 
        self.fc1  = nn.Linear(2*args.hs2, args.dense_out)
        self.relu = nn.ReLU()
        self.fc2  = nn.Linear(args.dense_out,1)
        self.sigm = nn.Sigmoid()
        # Dropouts
        self.drop_lstm = nn.Dropout(args.dp/2)
        self.drop_fc1 = nn.Dropout(args.dp)
    
    def forward(self,x):
        current_bs = len(x)
        embedded  = [self.embedding(torch.tensor(s).cuda()) for s in x]
        # LSTM 
        packed    = nn.utils.rnn.pack_sequence(embedded)
        output1,_ = self.lstm1(packed)
        output2,(h_n,c_n) = self.lstm2(output1)
        h_n2 = h_n.permute(1,0,2)
        h_n3 = torch.reshape(h_n2,(current_bs,-1))
        h_n3    = self.drop_lstm(h_n3)
        # Fully connected part 
        out = self.relu(self.fc1(h_n3))
        out = self.drop_fc1(out)
        out = self.sigm(self.fc2(out))
        return out


def restrict_len(t,max_len):
    dlen = len(t)
    if dlen >max_len:
        return t[:max_len]
    else:
        return t

def forward(model,x,y,loss_function,device):
    trn_x = x
    y = y.to(device)
    outs = model(trn_x)
    loss = loss_function(outs,y)
    return outs,loss
    
def make_batch(x,y,bs,max_len,is_shuffle=False,is_gen=False,is_test=False):
    lens = torch.tensor([len(i) for i in x])
    _,ix = torch.sort(lens,descending=True)
    if is_gen == False:
        x_sorted = [restrict_len(x[i],max_len) for i in ix]
        y_sorted = [y[i] for i in ix]
    else:
        x_sorted = [restrict_len(x[i],max_len) for i in range(len(x))]
        y_sorted = [y[i] for i in range(len(x))]
    
    batches  = []
    for i in range(0,len(x_sorted),bs):
        if is_test == False:
            batches.append((x_sorted[i:i+bs],
                            torch.tensor(y_sorted[i:i+bs]).reshape(len(y_sorted[i:i+bs]),1)))
        else:
            batches.append((x_sorted[i:i+bs],
                torch.tensor([-1]*bs).reshape(len(y_sorted[i:i+bs]),1)))
    if is_shuffle:
        random.shuffle(batches)
        
    return batches

def generate(emotion_name,model,data,labels,device):
    result = [] 
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        for (data_x,data_y) in data:
            data_y = data_y.to(device)
            outs   = model(data_x)
            result = result + [x[0] for x in outs.cpu().tolist()]
    return labels,result

def main_loop(model,optimizer,trn_batches,dev_batches,loss_function,device,args):
    # MAIN LOOP
    best_val_loss = float('inf')
    for epoch in range(args.epoch):
        total_loss = 0.0
        model.train()
        for (trn_x,trn_y) in trn_batches:
            model.zero_grad()
            outs,loss  = forward(model,trn_x,trn_y,loss_function,device)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_trn_loss = total_loss/len(trn_batches)
        # evaluation
        model.eval()
        with torch.no_grad():
            total_loss = 0.0
            for (dev_x,dev_y) in dev_batches:
                outs,loss = forward(model,dev_x,dev_y,loss_function,device)
                total_loss += loss .item()
            avg_dev_loss = total_loss/len(dev_batches)
            if avg_dev_loss < best_val_loss:
                best_val_loss = avg_dev_loss 
                torch.save(model.state_dict(), args.best_model_save_dir)
        print("Epoch:{} avg trn loss:{:.3f} avg dev loss:{:.3f}".format(epoch,avg_trn_loss,avg_dev_loss))
    return model,optimizer
