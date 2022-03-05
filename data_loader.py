import torch
import torch.utils.data as data
import re
from vocab import Vocabulary
import codecs
from sentence_transformers import models, SentenceTransformer



class TsvDataset(data.Dataset):
    """TSV custom dataset class, compatible with Dataloader"""

    def __init__(self, file_name, field_num=2):
        """

        :param file_name=path to the tsv file
        :param field_num=1 for covost_fr_en = field_num = fr, field_num = en
        """
        f = codecs.open(file_name, encoding='utf-8')
        self.field_num = field_num
        self.lines = f.readlines()
        self.sentence_labse = SentenceTransformer('/gpfsstore/rech/eie/upp27cx/labse', device='cuda')


    def __getitem__(self,index):
        """special python method for indexing a dict. dict[index]
        helper method to get annotation by id from coco.anns

        :param index: desired annotation id (key of annotation dict)

        return: (image, caption)
        """

        line = self.lines[index]
        fields = line.split('\t')
        txt = fields[self.field_num].lower()

        # tokenize captions
        caption = torch.Tensor([vocab(vocab.start_token())] +
                               [vocab(char) for char in txt] +
                               [vocab(vocab.end_token())])

        labSE = self.sentence_labse.encode(txt)

        return labSE, caption

    def __len__(self):
        return len(self.lines)

def collate_fn(data):
    """Create mini-batches of (image, caption)

    Custom collate_fn for torch.utils.data.DataLoader is necessary for patting captions

    :param data: list; (image, caption) tuples
            - image: tensor;    3 x 256 x 256
            - caption: tensor;  1 x length_caption

    Return: mini-batch
    :return images: tensor;     batch_size x 3 x 256 x 256
    :return padded_captions: tensor;    batch_size x length
    :return caption_lengths: list;      lenghths of actual captions (without padding)
    """

    # sort data by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge image tensors (stack)
    images = torch.stack(images, 0)

    # Merge captions
    caption_lengths = [len(caption) for caption in captions]

    # zero-matrix num_captions x caption_max_length
    padded_captions = torch.zeros(len(captions), max(caption_lengths)).long()

    # fill the zero-matrix with captions. the remaining zeros are padding
    for ix, caption in enumerate(captions):
        end = caption_lengths[ix]
        padded_captions[ix, :end] = caption[:end]
    return images, padded_captions, caption_lengths



def get_basic_loader(file_name, field_num, vocab, batch_size=32, shuffle=True, num_workers=2):
    """
    Returns torch.utils.data.DataLoader for custom coco dataset.
    :param dir_path:
    :param ann_path:
    :param vocab:
    :param transform:
    :param batch_size:
    :param shuffle:
    :param num_workers:
    :returns:
    """
    datas = TsvDataset(file_name, field_num)

    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for every iteration.
    # images: tensor of shape (batch_size, 3, 224, 224).
    # captions: tensor of shape (batch_size, padded_length).
    # lengths: list indicating valid length for each caption. length is (batch_size).
    data_loader = data.DataLoader(dataset=datas, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=num_workers,collate_fn=collate_fn)
    return data_loader
