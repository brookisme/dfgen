## DFGen 

**Keras Image Generator from Dataframes**

Creates generator from csv or dataframe.
Optional Features:

1. convert "tag" list to binary valued label vector for predictions
2. save to train/test split files
3. easy configuration with [yaml](#yaml) file

---

##### USAGE

In the exampe below we have used the `dfg_config.yaml` file located [here](https://github.com/brookisme/dfgen/blob/master/example.dfg_config.yaml).

```bash
>>> from dfgen import DFGen
>>> gen=DFGen(csv_file='train.csv',csv_sep=',',image_ext='tif')
>>> gen.size
40479
>>> gen.dataframe.head(2)
        image_name                                       tags  \
16452  train_16452  agriculture clear habitation primary road   
20043  train_20043                              clear primary   

                                                  labels  \
16452  [1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, ...   
20043  [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...   

                        dfg_paths  
16452  images/tif/train_16452.tif  
20043  images/tif/train_20043.tif  

#
# REQUIRE_LABEL:
#
>>> gen.require_label('blow_down',70)
>>> gen.size
140
>>> gen.tags
['primary', 'clear', 'agriculture', 'road', 'water', 'partly_cloudy', 'cultivation', 'habitation', 'haze', 'cloudy', 'bare_ground', 'selective_logging', 'artisinal_mine', 'blooming', 'slash_burn', 'conventional_mine', 'blow_down']
>>> gen.dataframe.sample(2)
      image_name                                             tags  \
55   train_23025   blow_down clear cultivation habitation primary   
101  train_20618                        clear cultivation primary   

                                                labels  \
55   [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, ...   
101  [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, ...   

                      dfg_paths  
55   images/tif/train_23025.tif  
101  images/tif/train_20618.tif  

#
# REQUIRE_LABEL: reduce_to_others=True
#
>>> gen.require_label('blow_down',70,reduce_to_others=True)
>>> gen.size
140
>>> gen.tags
['blow_down', 'others']
>>> gen.dataframe.sample(2)
      image_name                                         tags  labels  \
12   train_38607  agriculture blow_down partly_cloudy primary  [1, 1]   
24   train_31495            blow_down clear primary blow_down  [1, 1]   

                      dfg_paths  
12   images/tif/train_38607.tif  
109  images/tif/train_10679.tif  


#
# COMBINING REQUIRE LABELs
#
>>> from dfgen import DFGen
>>> gen=DFGen(csv_file='train.csv',csv_sep=',',image_ext='tif')
>>> >>> gen.size
183

# You can also fetch the rows with specific tags
>>> gen.dataframe_with_tags('blow_down','cultivation').size
32
>>> gen.dataframe_with_tags('blow_down','cultivation').head(2)
        image_name                                               tags  \
25950  train_25950  agriculture blooming blow_down clear cultivati...   
9961    train_9961    agriculture blow_down clear cultivation primary   

                                                  labels  \
25950  [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, ...   
9961   [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, ...   

                        dfg_paths  
25950  images/tif/train_25950.tif  
9961    images/tif/train_9961.tif  

# RequireLabel and check percentages
>>> gen.require_label('blow_down',10)
>>> gen.dataframe_with_tags('blow_down').shape[0]/gen.size
0.1
>>> gen.require_label('cultivation',60)
>>> gen.dataframe_with_tags('cultivation').shape[0]/gen.size
0.6010928961748634

# NOTE: The second require label effect the first.  
#       We no longer have exactly 10% blow_down.
>>> gen.dataframe_with_tags('blow_down').shape[0]/gen.size
0.07650273224043716




```


---

##### COMMENT-DOCS

```
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
    
    ...

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
        ...


    def dataframe_with_tags(self,*tags):
        """ return dataframe rows containing certain tags
            Args: strings of tag names
                ie. gen.dataframe_with_tags('blow_down','clear')
        """
        ...


    def save(self,path,split_path=None,split=0.2,sep=None):
        """ save dataframe to csv(s)

            usually save after processing (ie: tags->labels and/or require_label),
            so you wont need to process again.
            
            if split_path and split: 
                - split dataframe into 2 csvs (path and save path)
                - if split is int: split = number of lines in split_csv
                  else: split = % of full dataframe
        """
        ...
```


---


##### EXAMPLE CONFIG (in directory with .py or ipynb file)
<a name='yaml'></a>

[dfg_config.yaml](https://github.com/brookisme/dfgen/blob/master/example.dfg_config.yaml)

```
# COLUMN NAMES
image_column: image_name
label_column: labels
tags_column: tags

# IMAGE DIR BY EXT
image_dirs: 
  tif: images/tif
  jpg: images/jpg

# BACKUP IMAGE DIR
image_dir: images/other

# TAGS
tags:
    - primary
    - clear
    - agriculture
    - road
    - water
    - partly_cloudy
    - cultivation
    - habitation
    - haze
    - cloudy
    - bare_ground
    - selective_logging
    - artisinal_mine
    - blooming
    - slash_burn
    - conventional_mine
    - blow_down

```