# MIT License

# Copyright (c) 2018 the NJUNMT-pytorch authors.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import collections
import mmap
from typing import List
from typing import Union
import numpy as np

from .vocabulary import Vocabulary

__all__ = [
    'TextLineDataset',
    'ZipDataset',
    'QEHterDataset',
    'QETagsDataset'
]


def get_num_of_lines_before(filename):
    """Get number of lines of a file."""
    f = open(filename, "r+")
    buf = mmap.mmap(f.fileno(), 0)
    lines = 0
    readline = buf.readline
    while readline():
        lines += 1
    f.close()
    return lines


def get_num_of_lines(filename):
    """Get number of lines of a file."""
    with open(filename,'r',encoding='utf-8') as f:
        length = len(f.readlines())
    return length


class Record(object):
    """
    ```Record``` is one sample of a ```Dataset```. It has three attributions: ```data```, ```key``` and ```n_fields```.

    ```data``` is the actual data format of one sample. It can be a single field or more.
    ```key``` is used in bucketing, the larger of which means the size of the data.
    ```
    """
    __slots__ = ("fields", "size")

    def __init__(self, *fields, size):
        self.fields = fields
        self.size = size

    @property
    def n_fields(self):
        return len(self.fields)

def zip_records(*records: Record):
    """
    Combine several records into one single record. The key of the new record is the
    maximum of previous keys.
    """
    new_fields = ()
    sizes = []

    for r in records:
        new_fields += r.fields
        sizes.append(r.size)

    return Record(*new_fields, size=max(sizes))


class Dataset(object):
    """
    In ```Dataset``` object, you can define how to read samples from different formats of
    raw data, and how to organize these samples. Each time the ```Dataset``` return one record.

    There are some things you need to override:
        - In ```n_fields``` you should define how many fields in one sample.
        - In ```__len__``` you should define the capacity of your dataset.
        - In ```_data_iter``` you should define how to read your data, using shuffle or not.
        - In ```_apply``` you should define how to transform your raw data into some kind of format that can be
        computation-friendly. Must wrap the return value in a ```Record```， or return a ```None``` if this sample
        should not be output.
    """

    def __init__(self):

        self._size = None
        self._data_path = None

    def __len__(self):

        return self._size

    def __getitem__(self, index):
        with open(self._data_path, 'r', encoding='utf-8') as f:
            line = f.readlines()[index]
            line = self._apply(line, record=False)
            return line

        # return self.read(index)

    def showitem(self, index):
        with open(self._data_path, 'r', encoding='utf-8') as f:
            line = f.readlines()[index]
            return line
    
    def set_size(self, size):

        self._size = size

    def _apply(self, *lines, recore=True) -> Union[Record, None]:
        """ Do some processing on the raw input of the dataset.

        Return ```None``` when you don't want to output this line.

        Args:
            lines: A tuple representing one line of the dataset, where ```len(lines) == self.n_fields```

        Returns:
            A tuple representing the processed output of one line, whose length equals ```self.n_fields```
        """
        raise NotImplementedError


    def _data_iter(self):

        raise NotImplementedError
    
    def read(self, index=None):

        f_handles = self._data_iter()   # 一个打开的文件

        if not isinstance(f_handles, collections.Sequence):
            f_handles = [f_handles]

        count = 0
        

        for lines in zip(*f_handles):

            record = self._apply(*lines)
            """
            print("******************************")
            print(f_handles)
            print(zip(*f_handles))
            print(lines)
            print(record)
            """

            if record is not None:
                yield record
                count += 1

        [f.close() for f in f_handles]

        self.set_size(count)

class QETagsDataset(Dataset):
    """
    ```QETagsDataset``` is one kind of dataset each line of which is one sample. There is only one field each line.
    """

    def __init__(self,
                 data_path,
                 max_len=-1,
                 ):
        super(QETagsDataset, self).__init__()

        self._data_path = data_path
        self._max_len = max_len
        self.set_size(get_num_of_lines(self._data_path))

    #def __getitem__(self, id):
    #    pass
        # return open(self._data_path)

    def _data_iter(self):
        return open(self._data_path)

    def _apply(self, line, record=True):
        """
        Process one line
        :type line: str
        """
        line = line.strip().split(' ')
        new_line = []
        for each in line:
            if each == 'OK':
                new_line.append(1.0)
            else:
                new_line.append(2.0)
        line = new_line
        if 0 < self._max_len < len(line):
            return None

        if record == False:
            return [line]

        return Record(line, size=len(line))



class QEHterDataset(Dataset):
    """
    ```TextDataset``` is one kind of dataset each line of which is one sample. There is only one field each line.
    """

    def __init__(self,
                 data_path
                 ):
        super(QEHterDataset, self).__init__()

        self._data_path = data_path
        self.set_size(get_num_of_lines(self._data_path))

    def _data_iter(self):
        return open(self._data_path)

    def _apply(self, line, record=True):
        """
        Process one line

        :type line: str
        """
        line = [np.float32(line.strip().split())]
        if record == False:
            return [line]
        return Record(line, size=len(line))


class QEAlignDataset(Dataset):
    """
    ```TextDataset``` is one kind of dataset each line of which is one sample. There is only one field each line.
    """

    def __init__(self,
                 data_path
                 ):
        super(QEAlignDataset, self).__init__()

        self._data_path = data_path
        self.set_size(get_num_of_lines(self._data_path))

    def _data_iter(self):
        return open(self._data_path)

    def _apply(self, line, record=True):
        """
        Process one line

        :type line: str
        """
        align_info = dict()
        line = line.strip('\n').split()
        for aa in line:
            id_mt, id_pe = aa.split('-')
            id_mt = int(id_mt)
            id_pe = int(id_pe)
            align_info[id_mt] = id_pe

        if record == False:
            return align_info

        return Record(align_info, size=len(align_info))


class TextLineDataset(Dataset):
    """
    ```TextDataset``` is one kind of dataset each line of which is one sample. There is only one field each line.
    """

    def __init__(self,
                 data_path,
                 vocabulary,
                 max_len=-1,
                 ):
        super(TextLineDataset, self).__init__()

        self._data_path = data_path
        self._vocab = vocabulary  # type: Vocabulary
        self._max_len = max_len

        self.set_size(get_num_of_lines(self._data_path))

    #def __getitem__(self, id):


    def _data_iter(self):
        return open(self._data_path)

    def _apply(self, line, record=True):
        """
        Process one line

        :type line: str
        """
        line = self._vocab.sent2ids(line)

        if 0 < self._max_len < len(line):
            return None

        if record == False:
            return [line]

        return Record(line, size=len(line))


class TextLineStrDataset(Dataset):
    """
    ```TextDataset``` is one kind of dataset each line of which is one sample. There is only one field each line.
    """

    def __init__(self,
                 data_path,
                 max_len=-1,
                 ):
        super(TextLineStrDataset, self).__init__()

        self._data_path = data_path
        self._max_len = max_len

        self.set_size(get_num_of_lines(self._data_path))

    #def __getitem__(self, id):


    def _data_iter(self):
        return open(self._data_path)

    def _apply(self, line, record=True):
        """
        Process one line

        :type line: str
        """
        line = line.strip('\n').split()

        return Record(line, size=len(line))


class ZipDataset(Dataset):
    """
    ```ZipDataset``` is a kind of dataset which is the combination of several datasets. The same line of all
    the datasets consist on sample. This is very useful to build dataset such as parallel corpus in machine
    translation.
    """

    def __init__(self, *datasets):
        """
        """
        super(ZipDataset, self).__init__()

        self.datasets = datasets  # type: List[Dataset]

        self.set_size(len(self.datasets[0]))
    
    def __getitem__(self, id):
        
        return [d[id] for d in self.datasets]

    def showitem(self, id):
        
        return [d.showitem(id) for d in self.datasets]

    def _data_iter(self):

        return [d._data_iter() for d in self.datasets]

    def _apply(self, *lines: str) -> Union[Record, None]:
        """
        :type dataset: TextDataset
        """

        records = [d._apply(l) for d, l in zip(self.datasets, lines)]

        if any([r is None for r in records]):
            return None
        else:
            return zip_records(*records)
