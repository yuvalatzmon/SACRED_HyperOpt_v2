An example for integrating a general machine learning training script with 
[SACRED](https://github.com/IDSIA/sacred) experimental framework,
and [HyperOpt](https://github.com/hyperopt/hyperopt) (Distributed Asynchronous Hyperparameter Optimization).

Below you can also find full installation and usage instructions.

# Features
* Random order grid search, which is easy to config and manage, combining HyperOpt and SACRED. 
* Grid-search ranges support logarithmic, linear and categorical ranges.  
* Very easy to scale across machines and clusters. 
* Avoids running the same experiment twice.
* Experiments are logged on MongoDB with SACRED experimental framework.
* Example notebook for accessing results saved on MongoDB using PANDAS DataFrame.
* An example on how to quickly wrap a commandline based python training script.

# Project files
`mnist_keras.py` is a script to train mnist from commandline.

`sacred_wrapper.py` is a SACRED-HyperOpt wrapper for mnist_keras.

`hyperopt_search.py` is a distributed scheduler for hyper-params optimization.

`hopt_sacred.py` contains core classes and functions for this project 

`mongo_queries.ipynb` is a usage example for listing results saved by MongoDB, but 
using PANDAS DataFrame API. It also demonstrates manipulating (saving, loading, 
deleting) experiment artifacts (model files) using the MongoDB filesystem (GridFS).


# Setup the anaconda environment

    yes | conda create -n my_env python=3.6
    conda activate my_env
    yes | conda install -c anaconda tensorflow-gpu=1.9.0
    yes | pip install pymongo
    yes | pip install sacred hyperopt h5py
    yes | conda install pandas matplotlib ipython jupyter nb_conda
    yes | conda install cython 
    yes | pip install setuptools==39.1.0 # downgrade to meet tf 1.9 req
    yes | pip install networkx==1.11 # downgrade to meet hyperopt req
    
    # install sacredboard (optional)
    yes | pip install https://github.com/chovanecm/sacredboard/archive/develop.zip

    # Other common libraries I use, not required by this example
    yes | pip install GitPython markdown-editor 

# Installation instructions for MongoDB (v4.0.1) on RHEL/CENTOS, without root privileges
Both SACRED and HyperOpt require a MongoDB host. Installation instruction are based on [this](https://docs.mongodb.com/manual/tutorial/install-mongodb-on-red-hat/) page.

Execute the command below and also add it to your `~/.bashrc` file

    # Important: write the full path. Don't use ~/<...>, instead you can use ${HOME}/<...> .
    export MONGO_DIR=<your full path to MongoDB> 

Download and extract MongoDB tarball

    mkdir $MONGO_DIR
    cd $MONGO_DIR
    mkdir data
    mkdir log
    wget https://fastdl.mongodb.org/linux/mongodb-linux-x86_64-rhel70-4.0.1.tgz
    tar -zxvf mongodb-linux-x86_64-rhel70-4.0.1.tgz
    rm mongodb-linux-x86_64-rhel70-4.0.1.tgz

Execute the commands below and also add them to your `~/.bashrc` file

    export PATH=$MONGO_DIR/mongodb-linux-x86_64-rhel70-4.0.1/bin:$PATH

    # Port forwarding function
    # Usage example: pfwd hostname {6000..6009}
    function pfwd {
    for i in ${@:2}
    do
      echo Forwarding port $i
      ssh -N -L $i:localhost:$i $1 &
    done  
    }    

# Running Experiments

### 1. Start MongoDB server
Login to a mongo host machine in `<mongo_host_address>`

	mongod --fork --dbpath=$MONGO_DIR/data --logpath=$MONGO_DIR/log/mongo.log --logappend

Check that mongo server is running by `cat $MONGO_DIR/log/mongo.log` and verifying that the output contains the line ` [initandlisten] waiting for connections on port 27017`

### 2. Execute an experiment with default arguments on a client machine

	# Login to client machine
	ssh <client_address>
    
    # Forward MongoDB port to mongo host machine
    pfwd <mongo_host_address> 27017
    
	# Activate anaconda environment
    conda activate my_env

	# Execute experiment
    cd <project path>    
    python sacred_wrapper.py
    
Monitor results with [sacred board](https://github.com/chovanecm/sacredboard) (optional) 

	sacredboard -m MNISTdebug

### 3. Execute hyper-params random order grid-search on a distributed system (a cluster of machines)

For every client machine, do the following:

	# Login to client machine
	ssh <client_address>
    
    # Forward MongoDB port to mongo host machine
    pfwd <mongo_host_address> 27017
    
	# Activate anaconda environment
    conda activate my_env
	
    cd <project path>    
    
Login to client #1, run the commands above, and execute the hyper-params scheduler:
	
    python hyperopt_search.py 

For every other client machines (a "worker" machine), execute the worker script:

	export GPU_ID=<gpu_id> # select gpu id
    PYTHONPATH="./" CUDA_VISIBLE_DEVICES=$GPU_ID hyperopt-mongo-worker --mongo=localhost:27017/hyperopt_mnist --poll-interval=1 --workdir=`mktemp -u -p /tmp/hyperopt/`

## NOTE
This is an extension of my earlier repo https://github.com/yuvalatzmon/SACRED_HYPEROPT_Example, which demonstrated a more minimal example for integrating SACRED and hyperopt.


# References

1. https://www.hhllcks.de/blog/2018/5/4/version-your-machine-learning-models-with-sacred

2. https://github.com/IDSIA/sacred/issues/82#issuecomment-364067522

3. https://docs.mongodb.com/manual/tutorial/install-mongodb-on-red-hat/

4. https://stackoverflow.com/a/43165464/2476373 

5. https://github.com/IDSIA/sacred/issues/350

6. https://github.com/IDSIA/sacred/issues/351
