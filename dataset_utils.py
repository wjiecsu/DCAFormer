from   torch.utils.data import Dataset
import torch
import numpy  as np

class myDataset(Dataset):
      def __init__(self,data_list,data_label,seq_len):
            # 这里按照习惯的格式输入   [1,Batch_size, 2,seq_len, 3,feature_num],后面做permute(1,0,2)转换          
            X_3d =[]
            y_3d =[]
            for i in range(data_label.shape[0]-seq_len):
                X_3d.append(data_list[i:i+seq_len,:])
                y_3d.append(data_label[i+seq_len])#取第二天样本标签作为整个样本的标签
            self.data_list =np.stack(X_3d,0)
            self.data_label=np.stack(y_3d,0) 
               
      def __len__(self):
          return(self.data_list.shape[0])#为返回最新数据长度 
       
      #-告诉它一个样本是什么样的
      def __getitem__(self,index):  
          data   =self.data_list [index]
          label  =self.data_label[index]
          return [torch.Tensor(data).type(torch.FloatTensor),torch.from_numpy(np.array(label)).type(torch.FloatTensor)]      

def inversenormlization(max,min,data):
    return data*(max-min)+min