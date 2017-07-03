## DFGen 

**Keras Image Generator from Dataframes**

Creates generator from csv or dataframe.
Optional Features:

1. convert "tag" list to binary valued label vector for predictions
2. save to train/test split files
3. easy configuration with [yaml](#yaml) file

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


##### EXAMPLE CONFIG (in project root dir)
<a name='#yaml'></a>

**[dfg_config.yaml](https://github.com/brookisme/dfgen/blob/master/example.dfg_config.yaml)***

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