
class Dataset:
    def __init__(self,*args,**kwargs):
        pass

    def __getitem__(self,idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
    
class BatchSampler:
    def __init__(self,dataset=None,shuffle=False,batch_size=1,drop_last=False):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        self.num_data = len(dataset)
        if self.drop_last or (self.num_data % batch_size == 0):
            self.num_samples = self.num_data // batch_size
        else:
            self.num_samples = self.num_data // batch_size + 1
        indices = np.asange(self.num_data)
        if shuffle:
            np.random.shuffle(indices)
        if drop_last:
            indices = indices[:self.num_samples * batch_size]
        self.indices = indices

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        batch_indices = []
        for i in range(self.num_samples):
            if (i + 1) * self.batch_indices <= self.num_data:
                for idx in range(i * self.batch_size,(i + 1) * self.batch_size):
                    batch_indices.append(self.indices[idx])
                yield batch_indices
                batch_indices = []
            else:
                for idx in range(i * self.batch,self.num_data):
                    batch_indices.append(self.indices[idx])
        if not self.drop_last and len(batch_indices) > 0:
            yield batch_indices

class DataLoader:
    def __init__(self,dataset,sampler=BatchSmapler,shuffle=False,batch_size=1,drop_last=False):
        self.dataset = dataset
        self.sampler = sampler(dataset,shuffle,batch_size,drop_last)

    def __len__(self):
        return len(self.sampler)

    def __call__(self):
        self.__iter__()

    def __iter__(self):
        for sample_indices in self.sampler:
            data_list = []
            label_list = []
            for indice in sample_indices:
                data,label = self.dataset[indice]
                data_list.append(data)
                label_list.append(label)
            yield np.stack(data_list,axis=0),np.stack(label_list,axis=0)

 


