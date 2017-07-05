import os
import yaml
from skimage import io
from sklearn.utils import shuffle
import pandas as pd
import numpy as np


CSV_SEP=' '
USER_CONFIG='./dfg_config.yaml'
PATH_COLUMN='paths'
OTHERS_NAME='others'
ERROR_REQUIRED_COLUMNS='ERROR[DFGen]: image and label column are requrired.'
ERROR_TAGS_NOT_SET='ERROR[DFGEN]: reduce by tag requires tags be set'

class DFGen():
    """ CREATES GENERATOR FROM DATAFRAME
        
        create generator from existing dataframe or from a csv
        
        Methods:
            .require_label: ensure a min percentage of a particular label
            .save: save processed csv to csv or as train/test-split csvs
            .__next__: generator method, batchwise return tuple of (images,labels)

        Args:
            * image_column (column with image path or name) is required
            * label_column column with label "vectors" is required 
                - if the label_column already exists the dataframe will contain the labels
                - if the label_column does not exsit and both tags and tags_to_labels_column
                  are specified the tags will be converted to binary valued vectors
            * tags: optional list of tags in corresponding to places in the label vectors
            * tags_to_labels_column: name of a column that contain a space seperated 
                string of tags. these strings will be converted to the binary label vectors
            * image_dir: root path for image_paths given in "image_column"
            * image_ext:
                - append to image_column values when loading images
                - if using dfg_config file image_ext can determine image_dir
            * lambda_func: function that acts on image data before returned to user
            * batch_size: batch_size
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
        self.tags=tags or self._default('tags')
        self.batch_size=batch_size or self._default('batch_size')
        self.lambda_func=lambda_func
        self._set_dataframe(csv_file,dataframe,csv_sep)
        self._set_columns(image_column,label_column,tags_to_labels_column)
        self._init_labels()
        self._set_image_dir_and_ext(image_dir,image_ext)
        self._reset_generator()


    def dataframe_with_tags(self,*tags):
        """ return dataframe rows containing certain tags
            Args: strings of tag names
                ie. gen.dataframe_with_tags('blow_down','clear')
        """
        tags_exist=self.dataframe[self.tags_column].apply(
            lambda row_tags: self._has_tags(row_tags,tags))
        return self.dataframe[tags_exist]


    def require_label(self,label_index_or_tag,pct,exact=False,reduce_to_others=False):
        """
            Warning: Ordering matters

                .require_label(1,40)
                .require_label(2,20)
            
            may not equal:

                .require_label(2,20)
                .require_label(1,40)

            Args:
                * label_index_or_tag: <int|tag>
                    - <int>(label_index): index of the label of interest
                    - <str>(tag): if "tags": the name of the tag of interest
                * pct: <int:0-100> percentage required for label
                * exact:
                    if False and there is the label already has >= pct of dataset
                    return full-dataset
                    else: remove data so that label is pct of dataset
                * reduce to others.  
                    return labels as 2 vectors [label,others]
        """
        if isinstance(label_index_or_tag,str):
            label_index=self._tag_index(label_index_or_tag)
        else: 
            label_index=label_index_or_tag
        has_label_test=self.dataframe[self.label_column].apply(
            lambda v: v[label_index]==1)
        label_df=self.dataframe[has_label_test]
        label_size=label_df.shape[0]
        full_pct=label_size/self.size
        if (full_pct<pct) or exact:
            others_df=self.dataframe[~has_label_test]
            others_size=int(label_size*((100/pct)-1))
            others_df=others_df.sample(others_size)
            self.dataframe=pd.concat(
                [label_df,others_df],
                ignore_index=True).sample(frac=1)
        if reduce_to_others:
            self.reduce_columns(label_index,others=reduce_to_others)
        else:
            self._reset_generator()


    def reduce_columns(self,*indices_or_tags,others='others'):
        """ Keep passed columns and optional "others"

            Usage:
                gen.reduce_to_others('blow_down','cultivation')

            Args:
                * str or int arguments: label indices or tag names
                * others: 
                    - if falsey: do not include "others column"
                    - else:
                        include "others"
                        - if others arg is <str>: use others arg as column name
                        - else: use "others" as column name
        """
        if isinstance(indices_or_tags[0],str):
            label_indices=list(map(self._tag_index,indices_or_tags))
        else: 
            label_indices=indices_or_tags
        print('RC',label_indices)
        self.dataframe[self.label_column]=self.dataframe[self.label_column].apply(
            lambda label: self._reduce_label(label,label_indices,others))
        if self.tags:
            self.tags=list(map(lambda idx: self.tags[idx],label_indices))
            if others:
                if not isinstance(others,str): others='others'
                self.tags.append(others)
        self._reset_generator()


    def save(self,path,split_path=None,split=0.2,sep=None):
        """ save dataframe to csv(s)

            usually save after processing (ie: tags->labels and/or require_label),
            so you wont need to process again.
            
            if split_path and split: 
                - split dataframe into 2 csvs (path and save path)
                - if split is int: split = number of lines in split_csv
                  else: split = % of full dataframe
        """
        if split_path and split: 
            if isinstance(split,int): split_size=split
            else: split_size=int(self.size*split)
            self.dataframe=self.dataframe.sample(frac=1)
            split_df=self.dataframe[:split_size]
            df=self.dataframe[split_size:]
            self._save_df(split_df,split_path,sep)
            self._save_df(df,path,sep)
        else:
            self._save_df(self.dataframe,path,sep)


    def __next__(self):
        """ batchwise return tuple of (images,labels)
        """        
        start=self.batch_index*self.batch_size
        end=start+self.batch_size
        if (end>=self.size):
            self.batch_index=0
            self.labels, self.paths = shuffle(self.labels,self.paths)
        batch_labels=self.labels[start:end]
        batch_paths=self.paths[start:end]
        batch_imgs=[self._img_data(img) for img in batch_paths]
        self.batch_index+=1
        return np.array(batch_imgs),np.array(batch_labels)
    
    
    #
    # INTERNAL METHODS
    #
    def _init_properties(self):
        self.image_dir=None
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


    def _save_df(self,df,path,sep):
        df.to_csv(path,index=False,sep=sep or self.csv_sep)


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
                - if ext: try default by ext and image_dirs 
                - else if param: image_dir=param
                - or: default
        """
        self.image_ext=image_ext or self._default('image_ext')
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
            df=pd.read_csv(file_path,sep=self.csv_sep)
        self.size=df.shape[0]
        self.dataframe=df



    def _set_columns(self,image_column,label_column,tags_column):
        """ set image and label column
        """
        self.image_column=image_column or self._default('image_column')
        self.label_column=label_column or self._default('label_column')
        self.tags_column=tags_column or self._default('tags_column')
        if not (self.image_column and self.label_column):
            raise ValueError(ERROR_REQUIRED_COLUMNS)



    def _init_labels(self):
        """
            if tags and tags column:
                * if tags_column and label column does not exist create label column
                * else: ensure labels are lists
        """
        if self.tags_column and (self.label_column not in self.dataframe.columns):
            self.dataframe[self.label_column]=self.dataframe[self.tags_column].apply(
                self._tags_to_vec)
        else:
            self.dataframe[self.label_column]=self.dataframe[self.label_column].apply(
                self._to_list)


    def _to_list(self,str_list):
        """ Convert a list in string form to a list
            We must type check since:
                - if dataframe loaded from CSV vec will be a string
                - if dataframe created directly vec will be list
        """
        if type(str_list) is str:
            return list(eval(str_list))
        else:
            return str_list


    def _tag_index(self,tag):
        """
            return index for tag
        """
        if self.tags:
            return self.tags.index(tag)
        else:
            raise ValueError(ERROR_TAGS_NOT_SET)


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


    def _has_tags(self,row_tags,tags):
        """ check if tags are in row_tags
        """
        row_tags=row_tags.split(' ')
        return set(tags).issubset(row_tags)


    def _reduce_label(self,label,indices,others):
        """
            take vector and return the values at indices
            and a 1 or 0 if there are other nonzero values
        """
        label_values=list(map(lambda index: label.pop(index),indices))
        remainder=sum(label)
        if others: label_values.append(int(remainder>0))
        return label_values


    def _reset_generator(self):
        """ reset generator
            * reset batch index
            * shuffle dataframe
            * add PATH_COLUMN if it doesnt exist
            * set labels and paths
        """
        self.batch_index=0
        self.dataframe=self.dataframe.sample(frac=1)
        self.size=self.dataframe.shape[0]
        if PATH_COLUMN not in self.dataframe.columns:
            self.dataframe[PATH_COLUMN]=self.dataframe[self.image_column].apply(
                self._image_path_from_name)
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


