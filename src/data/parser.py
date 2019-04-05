import os
import pickle
import numpy as np
from scipy.io import wavfile
from sklearn.preprocessing import normalize
from collections import namedtuple


class ParserPCG:
    """
    ParserPCG is a container for loading phonocardiogram (PCG) data for the [2016 physionet
    challenge](http://physionet.org/challenge/2016). Raw wav files are parsed into
    features, class labels are extracted from header files and data is split into
    training and testing groups.
    """
    def __init__(self, basepath, endpath):
        self.basepath = basepath
        self.endpath = endpath
        self.class_name_to_id = {"normal": 0, "abnormal": 1}
        self.nclasses = len(self.class_name_to_id.keys())

        self.n_samples = 0

        # Truncating the length of each wav file to the
        # max file size (260000) (Note: this is bad
        # because we are filling with zeros all the files which len is bellow of 260000)
        self.embedding_size = 260000

        self.X = None
        self.y = None


    def initialize_wav_data(self):
        """
        Load the original wav files and extract features. Warning, this can take a
        while.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.load()
        
        # Saving data as npy in the end path
        self.save(self.endpath)

    def save(self, save_path):
        """
        Persist the PCG class to disk

        Parameters
        ----------
        save_path: str
            Location on disk to store the parsed PCG metadata

        Returns
        -------
        None

        """
        np.save(os.path.join(save_path, "X.npy"), self.X)
        np.save(os.path.join(save_path, "y.npy"), self.y)
        with open( os.path.join(save_path, "meta"), "w") as fout:
            pickle.dump((self.basepath, self.class_name_to_id, self.nclasses,
                         self.n_samples), fout)

    def load(self):
        """
        Loads physio 2016 challenge dataset from self.basepath by crawling the path.
        For each discovered wav file:

        * Attempt to parse the header file for class label
        * Attempt to load the wav file
        
        Returns
        -------
        None
        """

        # First pass to calculate number of samples
        # ensure each wav file has an associated and parsable
        # Header file
        wav_file_names = []
        class_labels = []
        for root, dirs, files in os.walk(self.basepath):
            # Ignore validation for now!
            if "validation" in root:
                continue
            for file in files:
                if file.endswith('.wav'):
                    try:
                        base_file_name = file.rstrip(".wav")
                        label_file_name = os.path.join(root, base_file_name + ".hea")

                        class_label = self.__parse_class_label(label_file_name)
                        class_labels.append(self.class_name_to_id[class_label])
                        wav_file_names.append(os.path.join(root, file))

                        self.n_samples += 1
                    except InvalidHeaderFileException as e:
                        print(e) 

        
        # Inicialize X zeros array 
        X = np.zeros([self.n_samples, self.embedding_size])

        for idx, wavfname in enumerate(wav_file_names):
            rate, wf = wavfile.read(wavfname)
            wf = normalize(wf.reshape(1, -1))

            wf = wf[0]

            if len(wf) > self.embedding_size:
                X[idx, :] = wf[:self.embedding_size]
            else:
                X[idx, :len(wf)] = wf

            idx += 1

        self.X = X

        class_labels = np.array(class_labels)

        # Map from dense to one hot
        self.y = np.eye(self.nclasses)[class_labels]

    def __parse_class_label(self, label_file_name):
        """
        Parses physio bank header files, where the class label
        is located in the last line of the file. An example header
        file could contain:

        f0112 1 2000 60864
        f0112.wav 16+44 1 16 0 0 0 0 PCG
        # Normal


        Parameters
        ----------
        label_file_name: str
            Path to a specific header file

        Returns
        -------
        class_label: str
            One of `normal` or `abnormal`
        """
        with open(label_file_name, 'r') as fin:
            header = fin.readlines()

        comments = [line for line in header if line.startswith("#")]
        if not len(comments) == 1:
            raise InvalidHeaderFileException("Invalid label file %s" % label_file_name)

        class_label = str(comments[0]).lstrip("#").rstrip("\r").strip().lower()

        if not class_label in self.class_name_to_id.keys():
            raise InvalidHeaderFileException("Invalid class label %s" % class_label)

        return class_label


class InvalidHeaderFileException(Exception):
    def __init__(self, *args, **kwargs):
        super(args, kwargs)
