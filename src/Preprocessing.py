#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2023 Dmitry Degtyar, Jan Ruhland, Dominik Heider

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import torch
import torchvision.transforms as transforms
from PIL import Image
from imblearn.over_sampling import RandomOverSampler as ROS
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np




class Preprocessor:
    def __init__(self, data):
        self.data=data
        
    def oversample(self):
        # Create over sampler, and over sample ONLY the TRAIN dataframe
        oversample = ROS(random_state=42)
        X = np.array(X_train.values.tolist()).squeeze()
        
        # reshape train dataset fro oversamler
        reshaped_X = X.reshape(X.shape[0], -1)
        
        # oversampling
        oversampled_X, oversampled_y = oversample.fit_resample(reshaped_X, y_train)
        
        # reshaping X back to the first dims
        new_X = oversampled_X.reshape(-1, 450, 450, 3)
        
        # creating series from 4d numpy array
        X_train = pd.Series([new_X[x, :, :, :] for x in range(new_X.shape[0])], dtype=object, name='image')
        
        # new oversampled labels as train labels
        y_train = pd.Series(oversampled_y)
        
        # new oversampled labels as train labels
        y_train = pd.Series(oversampled_y)
        
        y_val = pd.Series(y_val)
        
        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)
        val_df.to_pickle("validation_data.pickle")
        train_df.to_pickle("training_data.pickle")
        
    def setData(self, X_train, X_val, y_train, y_val):
        return 0
    
    
    
    
    
class HAMDataLoader(Dataset):

    def __init__(self, dataset, transforms=None, selective=True, OR=5, input_size=224):
        """

        Parameters
        ----------
        dataset : TYPE
            DESCRIPTION.
        transforms : TYPE, optional
            DESCRIPTION. The default is None.
        selective : TYPE, optional
            DESCRIPTION. The default is True.
        OR : TYPE, optional
            DESCRIPTION. The default is 5.
        input_size : TYPE, optional
            DESCRIPTION. The default is 224.

        Returns
        -------
        None.

        """
        self.df = dataset

        self.OR = OR  # the label of the over represented class
        self.tf = transforms  # the input list of transforms
        self.selective = selective  # flag to apply transform to under rep classes only
        self.input_size = input_size

    def __len__(self):
        return len(self.df)

    def to_categorical(y, num_classes=None, dtype="float32"):
        """
        

        Parameters
        ----------
        y : TYPE
            DESCRIPTION.
        num_classes : TYPE, optional
            DESCRIPTION. The default is None.
        dtype : TYPE, optional
            DESCRIPTION. The default is "float32".

        Returns
        -------
        categorical : TYPE
            DESCRIPTION.

        """
        y = np.array(y, dtype="int")
        input_shape = y.shape
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        y = y.ravel()
        if not num_classes:
            num_classes = np.max(y) + 1
        n = y.shape[0]
        categorical = np.zeros((n, num_classes), dtype=dtype)
        categorical[np.arange(n), y] = 1
        output_shape = input_shape + (num_classes,)
        categorical = np.reshape(categorical, output_shape)
        return categorical

    def __getitem__(self, idx):
        """
        

        Parameters
        ----------
        idx : TYPE
            DESCRIPTION.

        Returns
        -------
        dict
            DESCRIPTION.

        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pil_image = Image.fromarray(self.df.iloc[idx].loc['image'])
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize(self.input_size),
                                        transforms.Normalize(
                                            mean=[0.764, 0.538, 0.562],
                                            std=[0.138, 0.159, 0.177],
                                        )
                                        ])
        img_tensor = transform(pil_image)

        category = self.df.iloc[idx].loc['label']
        disease = torch.tensor(category, dtype=torch.long)

        # Applying transforms
        if self.tf != None:
            if self.selective == True:  # Can choose to NOT apply augmentation on over rep classes
                if torch.argmax(disease) != self.OR:
                    img_tensor = self.tf(img_tensor)
            else:  # Or just apply aug to ALL classes and samples
                img_tensor = self.tf(img_tensor)

        return {'image': img_tensor.double(), 'disease': disease}
        
        
        
        



if __name__ == "__main__":
    # We performed a binary transformation of the 
    # HAM10000 data set for faster processing
    # The transformed data can be accessed here: 
    df = pd.read_pickle('HAMdataframe.pickle') 
    
    # Splitting the data with given random state
    X_train, X_val, y_train, y_val = train_test_split(df.image, df.label, test_size=0.2, random_state=42, stratify=df.label)
    
    # Create over sampler, and over sample ONLY the TRAIN dataframe
    oversample = ROS(random_state=42)
    X = np.array(X_train.values.tolist()).squeeze()

    # reshape train dataset fro oversamler
    reshaped_X = X.reshape(X.shape[0], -1)
    
    # oversampling
    oversampled_X, oversampled_y = oversample.fit_resample(reshaped_X, y_train)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
