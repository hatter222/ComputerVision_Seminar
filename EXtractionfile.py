import tarfile

tf = tarfile.open("cifar-10-python.tar.gz")
tf.extractall()