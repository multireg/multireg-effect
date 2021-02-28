import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import random

class Anger2(nn.Module):
    # out_channel: # output channel of convolutions
    # dense_out :  1st dense layer output  dimension
    def __init__(self,embedding_matrix,out_channel,dense_out,n_gram_num,pos_num):
        super(Anger2, self).__init__()
        num_embeddings,embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding(num_embeddings,embedding_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix,
                                                          dtype=torch.float))
        self.pos_embedding = nn.Embedding(pos_num,24)
        self.conv2  = nn.Conv1d(24+embedding_dim,out_channel, kernel_size = 2)
        self.conv3  = nn.Conv1d(24+embedding_dim,out_channel, kernel_size = 3)
        self.conv4  = nn.Conv1d(24+embedding_dim,out_channel, kernel_size = 4)
        self.conv5  = nn.Conv1d(24+embedding_dim,out_channel, kernel_size = 5)
        self.conv6  = nn.Conv1d(24+embedding_dim,out_channel, kernel_size = 6) 
        # https://pytorch.org/docs/stable/nn.html#torch.nn.BatchNorm1d
        #self.dense1_bn = nn.BatchNorm1d(out_channel*n_gram_num)
        self.drop = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(out_channel*n_gram_num+3, dense_out)
        self.sg1 = nn.Sigmoid()
        self.fc2 = nn.Linear(dense_out, 1)
        self.sg2 = nn.Sigmoid()
         
    def forward(self,x,lex_x,pos_x):
        # x shape: BSxMAX_LEN
        embeds     = self.embedding(x) # BS(64) x MAX_LEN(80) x EMBED_SIZE(200)
        embeds_pos = self.pos_embedding(pos_x)

        embeds = torch.cat((embeds,embeds_pos),2)
        embeds = embeds.permute(0, 2, 1)

        #embeds_pos = embeds_pos.permute(0,2,1)
        #embeds = torch.cat((embeds,embeds_pos),1)
        
        o2 = self.conv2(embeds) # 64,200,79
        o2 = F.max_pool1d(o2,o2.shape[2])# 64,200,1
        o2 = torch.squeeze(o2) # 64,200

        o3 = self.conv3(embeds)
        o3 = torch.squeeze(F.max_pool1d(o3,o3.shape[2]))
        
        o4 = self.conv4(embeds)
        o4 = torch.squeeze(F.max_pool1d(o4,o4.shape[2]))
        
        o5 = self.conv5(embeds)
        o5 = torch.squeeze(F.max_pool1d(o5,o5.shape[2]))
 
        o6 = self.conv6(embeds)
        o6 = torch.squeeze(F.max_pool1d(o6,o6.shape[2]))
        if x.shape[0] == 1:
            concatenated = torch.cat((o2,o3,o4,o5,o6))
            concatenated = torch.reshape(concatenated,(1,len(concatenated)))
        else:
            concatenated = torch.cat((o2,o3,o4,o5,o6), 1) # 64x1000
        concatenated = torch.cat((concatenated,lex_x),1)
        #concatenated = self.dense1_bn(concatenated)
        out = self.drop(concatenated)
        out = self.sg1(self.fc1(out))
        out = self.sg2(self.fc2(out))
        return out

def make_batch(x,y,lex_x,pos_x,bs,max_len,stoi,ptoi,is_shuffle=False,is_gen=False,is_test=False):
    lens = torch.tensor([len(i) for i in x])
    _,ix = torch.sort(lens)
    if is_gen == False:
        x_sorted = [restrict_len(x[i],max_len,stoi) for i in ix]
        pos_sorted = [restrict_len(pos_x[i],max_len,ptoi) for i in ix]        
        y_sorted = [y[i] for i in ix]
        lex_sorted = [lex_x[i] for i in ix]
    else:
        x_sorted = [restrict_len(x[i],max_len,stoi) for i in range(len(x))]
        pos_sorted = [restrict_len(pos_x[i],max_len,ptoi) for i in range(len(x))]        
        y_sorted = [y[i] for i in range(len(x))]
        lex_sorted = [lex_x[i] for i in range(len(lex_x))]
    batches  = []
    for i in range(0,len(x_sorted),bs):
        batches.append((torch.tensor(x_sorted[i:i+bs]),
                        torch.tensor(y_sorted[i:i+bs]).reshape(len(y_sorted[i:i+bs]),1),
                        torch.tensor(lex_sorted[i:i+bs]),torch.tensor(pos_sorted[i:i+bs])))
    if is_shuffle:
        random.shuffle(batches)        
    return batches

def generate(emotion_name,model,data,labels,device):
    result = [] 
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        for (data_x,data_y,lex_x,pos_x) in data:
            data_x = data_x.to(device)
            data_y = data_y.to(device)
            lex_x = lex_x.to(device)
            pos_x = pos_x.to(device)
            outs   = model(data_x,lex_x,pos_x)
            result = result + [x[0] for x in outs.cpu().tolist()]
    return labels,result

def restrict_len(t,max_len,stoi):
    dlen = len(t)
    if dlen >max_len:
        return t[:max_len]
    else:
        diff = (max_len - dlen)/2
        d1 = math.floor(diff)
        d2 = math.ceil(diff)
        return t+(d1+d2)*[stoi['_pad_']]

def forward(model,x,y,lex_x,pos_x,loss_function,device):
    trn_x = x.to(device)
    y = y.to(device)
    lex_x = lex_x.to(device)
    pos_x = pos_x.to(device)
    outs = model(trn_x,lex_x,pos_x)
    loss = loss_function(outs,y)
    return outs,loss

def main_loop(model,optimizer,trn_batches,dev_batches,loss_function,device,args):
    # MAIN LOOP
    best_val_loss = float('inf')
    for epoch in range(args.epoch):
        total_loss = 0.0
        model.train()
        for (trn_x,trn_y,lex_sent,pos_x) in trn_batches:
            model.zero_grad()
            outs,loss  = forward(model,trn_x,trn_y,lex_sent,pos_x,loss_function,device)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_trn_loss = total_loss/len(trn_batches)
        # evaluation
        model.eval()
        with torch.no_grad():
            total_loss = 0.0
            for (dev_x,dev_y,lex_sent,pos_x) in dev_batches:
                outs,loss = forward(model,dev_x,dev_y,lex_sent,pos_x,loss_function,device)
                total_loss += loss .item()
            avg_dev_loss = total_loss/len(dev_batches)
            if avg_dev_loss < best_val_loss:
                best_val_loss = avg_dev_loss 
                torch.save(model.state_dict(), args.best_model_save_dir)
        print("Epoch:{} avg trn loss:{:.3f} avg dev loss:{:.3f}".format(epoch,avg_trn_loss,avg_dev_loss))
    return model,optimizer
