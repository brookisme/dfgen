import os
import yaml
from skimage import io
from sklearn.utils import shuffle
import pandas as pd
import numpy as np


CSV_SEP=' '
USER_CONFIG='./dfg_config.yaml'
PATH_COLUMN='dfg_paths'
ERROR_REQUIRED_COLUMNS='ERROR[DFGen]: both image and label column are requrired.'


class DFGen():
    """ CREATES GENERATOR FROM DATAFRAME
        * load df directly from a df or from a csv
        * ...
    """
    def __init__(
            self,
            csv_file=None,
            dataframe=None,
            image_column=None,
            label_column=None,
            tags_to_labels_column=None,
            tags=None,
            batch_size=None,
            image_dir=None,
            image_ext=None,
            lambda_func=False,
            csv_sep=None):
        self._init_properties()
        self._load_defaults()
        self._set_dataframe(csv_file,dataframe,csv_sep)
        self._set_columns(image_column,label_column)
        self._set_tags(tags,tags_to_labels_column)
        self._set_paths_and_labels()
        self._set_image_dir_and_ext(image_dir,image_ext)
        self.batch_size=batch_size or self._default('batch_size')
        self.lambda_func=lambda_func
        self.batch_index=0


    def save_df(self,path):
        """
            save processed-dataframe
            once saved can pass without 
            tags_to_labels_column
        """
        self.dataframe.to_csv(path,sep=self.csv_sep)


    def restrict(self,pct_list):
        """
            Args:
                pct_list: 
                    list corresponding to labels vector
                    on what percentage of each column to keep.
                    - int (% to keep)
                    - '*' keep any percentage
                    - '!' remove
            ??? what about 50% blowdown 50% not blowdown
        """
        pass


    def __next__(self):
        """
            batchwise return tuple of (images,labels)
        """        
        start=self.batch_index*self.batch_size
        end=start+self.batch_size
        if (end>=self.size):
            self.labels, self.paths = shuffle(self.labels,self.paths)
            self.batch_index=0
        batch_labels=self.labels[start:end]
        batch_paths=self.paths[start:end]
        batch_imgs=[self._img_data(img) for img in batch_paths]
        self.batch_index+=1
        return np.array(batch_imgs),np.array(batch_labels)
    
    
    #
    # INTERNAL METHODS
    #
    def _init_properties(self):
        self._image_dir=None
        self._lambda_func=None


    def _load_defaults(self):
        """ 
            if config file exsits: self._defaults=<dict-from-config>
            else: self._defaults={}
        """
        if os.path.isfile(USER_CONFIG):
            self._defaults=yaml.safe_load(open(USER_CONFIG))
        else:
            self._defaults={}


    def _default(self,key):
        """ safe get default
        """
        return self._defaults.get(key,None)


    def _img_data(self,path):
        """Read Data 
            if self.lambda_func: apply lambda_func

            Args:
                path: <str> path to image
        """
        img=io.imread(path)
        if self.lambda_func:
            return self.lambda_func(img)
        else:
            return img


    def _set_image_dir_and_ext(self,image_dir,image_ext):
        """Set image dir and image ext
            * image_ext = param or default
            * image_dir:
                - if param: image_dir=param
                - else: try default by ext and image_dirs
                - or: default
        """
        self.image_ext=image_ext or self._default('image_ext')
        if image_dir: self.image_dir=image_dir
        else:
            if self.image_ext:
                image_dirs=self._default('image_dirs')
                if image_dirs: 
                    self.image_dir=image_dirs.get(self.image_ext)
            if not self.image_dir:
                self.image_dir=self._default('image_dir')


    def _set_dataframe(self,file_path,df,csv_sep):
        """Set Data
            sets three instance properties:
                self.labels
                self.paths
                self.dataframe
            the paths and labels are pairwised shuffled
        """
        self.csv_sep=csv_sep or self._default('csv_sep') or CSV_SEP
        if file_path: 
            df=pd.read_csv(self.file_path,sep=self.csv_sep)
        self.size=df.shape[0]
        self.dataframe=df



    def _set_columns(self,image_column,label_column):
        """ set image and label column
        """
        self.image_column=image_column or self._default('image_column')
        self.label_column=label_column or self._default('label_column')
        if not (self.image_column and self.label_column):
            raise ValueError(ERROR_REQUIRED_COLUMNS)



    def _set_tags(self,tags,tag_column):
        """
            if tags and tags column:
                * set tag properties
                * if tags_column: create label column from tags
        """
        self.tags=tags
        if tag_column:
            self.tag_column=tag_column
            self.dataframe[self.label_column]=self.dataframe[self.tags_column].apply(
                self._tags_to_vec)


    def _tags_to_vec(self,tags):
        """ 
            - convert tag-list-string to a binary-valued label vector
            - list ordering given by the tags property
            Args:
                * tags: is a string containing space-seperated 
                  strings, the tags themselves. 
        """
        tags=tags.split(' ')
        return [int(label in tags) for label in self.tags]


    def _set_paths_and_labels(self):
        """
            - create PATH_COLUMN from image_column
            - set shuffled label/path pairs
        """
        self.dataframe[PATH_COLUMN]=self.dataframe[self.image_column].apply(self._image_path_from_name)
        labels=self.dataframe[self.label_column].values.tolist()
        paths=self.dataframe[PATH_COLUMN].values.tolist()
        self.labels, self.paths = shuffle(labels,paths)


    def _image_path_from_name(self,name):
        """ Get image path from image name
            - prepend image_dir if exists
            - append image_ext if exists
            Args:
                name: image name/path
        """
        parts=[part for part in [self.image_dir,name] if part is not None]
        image_path='/'.join(parts)
        if self.image_ext: 
            image_path='{}.{}'.format(image_path,self.image_ext)
        return image_path


