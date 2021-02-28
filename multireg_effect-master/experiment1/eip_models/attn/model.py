import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import random
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence

class  THU_NGN(nn.Module):
    def __init__(self,embedding_matrix,args):#hidden_dim,out_channel,dense_out,n_gram_num,dp):
        super(THU_NGN, self).__init__()
        num_embeddings,embedding_dim = embedding_matrix.shape
        self.max_len = args.max_len
        self.embedding = nn.Embedding(num_embeddings,embedding_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix,dtype=torch.float))
        self.embedding.weight.requires_grad = False        
        self.lstm = nn.LSTM(embedding_dim, args.hs,
                            num_layers=args.layer_num, bidirectional=args.bidirectional)
        
        self.attention_weight = nn.Linear(2*args.hs,1)       
        self.tanh = nn.Tanh()
        # CNN Weights
        self.conv3  = nn.Conv1d(2*args.hs,args.filter_number, kernel_size = 3)
        self.conv5  = nn.Conv1d(2*args.hs,args.filter_number, kernel_size = 5)
        self.conv7  = nn.Conv1d(2*args.hs,args.filter_number, kernel_size = 7)
        self.conv9  = nn.Conv1d(2*args.hs,args.filter_number, kernel_size = 9)
        # Dense Layer weights 
        self.fc1  = nn.Linear(args.filter_number*args.n_gram_num, args.dense_out)
        self.relu = nn.ReLU()
        self.fc2  = nn.Linear(args.dense_out, 1)
        self.sigm = nn.Sigmoid()
        # Dropouts
        self.drop_lstm = nn.Dropout(args.dp)
        self.drop_attn_weight = nn.Dropout(args.dp)
        self.drop_concat = nn.Dropout(args.dp)
        self.drop_fc1 = nn.Dropout(args.dp)
    
    def forward(self,x):
        current_bs = len(x)
        embedded  = [self.embedding(torch.tensor(s).cuda()) for s in x]
        packed    = nn.utils.rnn.pack_sequence(embedded)
        output,_  = self.lstm(packed)
        output,_  = pad_packed_sequence(output) # output shape (41,8,400) # max_seq_len,bs,2hidden_size
        output    = self.drop_lstm(output)
        # Attention 
        hs_values = []
        for i in range(current_bs):
            hidden_states   = output[:len(x[i]),i,:] # print Nx 2hs
            hs2             = self.tanh(hidden_states) 
            unnorm_attn     = self.attention_weight(hs2) # Nx1
            unnorm_attn     = self.drop_attn_weight(unnorm_attn)
            unnorm_attn_exp = torch.exp(unnorm_attn)
            context_vector  = unnorm_attn_exp/torch.sum(unnorm_attn_exp) # Nx1 
            r_i             = hidden_states * context_vector # Nx2hs
            pd   = nn.ZeroPad2d((0,0,0,self.max_len-len(x[i])))
            r_i2 = pd(r_i)
            r_i2 = torch.reshape(torch.t(r_i2),(1,-1,self.max_len)) # 1,2hs,N (1,2hs,50)
            hs_values.append(r_i2)
        # CNN
        cnn_input = torch.cat(hs_values,0)
        o3 = self.conv3(cnn_input)
        o3 = torch.squeeze(F.max_pool1d(o3,o3.shape[2]))
            
        o5 = self.conv5(cnn_input)
        o5 = torch.squeeze(F.max_pool1d(o5,o5.shape[2]))
            
        o7 = self.conv7(cnn_input)
        o7 = torch.squeeze(F.max_pool1d(o7,o7.shape[2]))
            
        o9 = self.conv9(cnn_input)
        o9 = torch.squeeze(F.max_pool1d(o9,o9.shape[2]))
        # DENSE LAYER
        if current_bs != 1:
            concatenated = torch.cat((o3,o5,o7,o9),1)     
        else:
            concatenated = torch.cat((o3,o5,o7,o9))
        out = self.drop_concat(concatenated)
        out = self.relu(self.fc1(out))
        out = self.drop_fc1(out)
        out = torch.reshape(self.sigm(self.fc2(out)),(-1,1))
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
