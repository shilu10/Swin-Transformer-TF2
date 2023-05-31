class DataLoader: 
    """
        Class, will be useful for creating the BYOL dataset or dataset for the DownStream task 
            like classification or segmentation.
        Methods:
            __download_data(scope: private)
            __normalize(scope: private)
            __preprocess_img(scope: private)
             __get_val_data(scope: private)
            get_dataset(scope: public)
        
        Property:
            dname(dtype: str)        : dataset name(supports cifar10, cifar100).
            augmentor(type           : byol augmentor instance/object.
            nval(type: int)          : Number of validation data needed, this will be created by splitting the testing
                                       data.
            resize_shape(dtype: int) : Resize shape, bcoz pretrained models, might have a different required shape. If Resize shape is 0
                                        no resize, will happen.
            normalize(dtype: bool)   : bool value, whether to normalize the data or not. 
    """
    
    def __init__(self, dname="cifar10", augmentor=None, nval=5000, resize_shape=0, normalize=True): 
        assert dname in ["cifar10", 'cifar100'], "dname should be either cifar10 or cifar100"
        assert nval <= 10_000, "ValueError: nval value should be <= 10_000"
        
        __train_data, __test_data = self.__download_data(dname)
        self.__train_X, self.__train_y = __train_data
        self.__train_X, self.__train_y = self.__train_X, self.__train_y
        self.__dtest_X, self.__dtest_y = __test_data 
        self.__dname = dname
        
        self.augmentor = augmentor
        self.__get_val_data(nval)
        self.resize_shape = resize_shape
        
        
        self.__normalize() if normalize else None
        
    def __len__(self): 
        return self.__train_X.shape[0] + self.__dtest_X.shape[0]
    
    def __repr__(self): 
        return f"Training Samples: {self.__train_X.shape[0]}, Testing Samples: {self.__dtest_X.shape[0]}"
    
    def __download_data(self, dname):
        """
            Downloads the data from the tensorflow website using the tensorflw.keras.load_data() method.
            Params:
                dname(type: Str): dataset name, it just supports two dataset cifar10 or cifar100
            Return(type(np.ndarray, np.ndarray))
                returns the training data and testing data
        """
        if dname == "cifar10": 
            train_data, test_data = tf.keras.datasets.cifar10.load_data()
        if dname == "cifar100": 
            train_data, test_data = tf.keras.datasets.cifar100.load_data()
            
        return train_data, test_data
    
    def __normalize(self): 
        """
            this method, will used to normalize the inputs.
        """
        self.__train_X = self.__train_X / 255.0
        self.__dtest_X = self.__dtest_X / 255.0
    
    def __preprocess_img(self, image, label, augment=False): 
        """
            this method, will be used by the get_byol_dataset methos, which does a convertion of 
            numpy data to tensorflow data.
            Params:
                image(type: np.ndarray): image data.
            Returns(type; (np.ndarray, np.ndarray))
                returns the two different augmented views of same image.
        """
        try: 
            image = tf.cast(image, tf.float32)
            if self.__dname == 'cifar10':
                label = tf.one_hot(label, 10)
            else:
                label = tf.one_hot(label, 100)
            
            if self.resize_shape > 0:
                image = tf.image.resize(image, (self.resize_shape, self.resize_shape))
         
            if self.augmentor and augment:
                image = self.paugmentor.augment(image)
           
            return image, label
        
        except Exception as err:
            return err
    
    def get_dataset(self, batch_size, dataset_type="train"):
        """
            this method, will gives the byol dataset, which is nothing but a tf.data.Dataset object.
            Params:
                batch_size(dtype: int)    : Batch Size.
                dataset_type(dtype: str)  : which type of dataset needed, (train, test or val)
                
            return(type: tf.data.Dataset)
                returns the tf.data.Dataset for intended dataset_type, by preprocessing and converting 
                the np data.
        """
        try:
            if dataset_type == "train":
                tensorflow_data = tf.data.Dataset.from_tensor_slices((self.__train_X, self.__train_y))
                tensorflow_data = (
                tensorflow_data
                    .shuffle(1024)
                    .map(lambda x, label: (self.__preprocess_img(x, label,  augment=True)),
                                                 num_parallel_calls=tf.data.AUTOTUNE).cache()
                    .batch(batch_size, drop_remainder=True)
                    .prefetch(tf.data.experimental.AUTOTUNE)
                )
                return tensorflow_data  
            
            if dataset_type == "test":
                tensorflow_data = tf.data.Dataset.from_tensor_slices((self.__test_X, self.__test_y))
                tensorflow_data = (
                tensorflow_data
                    .shuffle(1024)
                    .map(lambda x, label: (self.__preprocess_img(x, label, augment=False)),
                                                 num_parallel_calls=tf.data.AUTOTUNE).cache()
                    .batch(batch_size, drop_remainder=True)
                    .prefetch(tf.data.experimental.AUTOTUNE)
                )
                return tensorflow_data  
            
            if dataset_type == "val":
                tensorflow_data = tf.data.Dataset.from_tensor_slices((self.__val_X, self.__val_y))
                tensorflow_data = (
                tensorflow_data
                     .shuffle(1024)
                    .map(lambda x, label: (self.__preprocess_img(x, label, augment=False)),
                                                 num_parallel_calls=tf.data.AUTOTUNE).cache()
                    .batch(batch_size, drop_remainder=True)
                    .prefetch(tf.data.experimental.AUTOTUNE)
                )
                return tensorflow_data  
        
        except Exception as err:
            return err
    
   
    def __get_val_data(self, nval):
        """
            this method is used to create a validation data by randomly sampling from the testing data.
            Params:
                nval(dtype: Int); Number of validation data needed, rest of test_X.shape[0] - nval, will be 
                                  testing data size.
            returns(type; np.ndarray, np.ndarray):
                returns the testing and validation dataset.
        """
        try: 
            ind_arr = np.arange(10_000)
            val_inds = np.random.choice(ind_arr, nval, replace=False)
            test_inds = [i for i in ind_arr if not i in val_inds]

            self.__test_X, self.__test_y = self.__dtest_X[test_inds], self.__dtest_y[test_inds]
            self.__val_X, self.__val_y = self.__dtest_X[val_inds], self.__dtest_y[val_inds]
            
        except Exception as err:
            raise err    

