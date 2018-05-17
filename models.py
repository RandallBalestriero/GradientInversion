execfile('lasagne_tf.py')
execfile('utils.py')
import random

def onehot(n,k):
        z=zeros(n,dtype='float32')
        z[k]=1
        return z




	



class DNNClassifier(object):
	def __init__(self,input_shape,model_class,lr=0.0001,optimizer = tf.train.AdamOptimizer,l2=0,learn_beta=0,e=1):
		#setting = {base,pretrainlinear,}
		tf.reset_default_graph()
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		self.n_classes=  model_class.n_classes
		config.log_device_placement=True
		self.session = tf.Session(config=config)
		self.batch_size = input_shape[0]
		self.lr = lr
		opt = adam(lr)
		with tf.device('/device:GPU:0'):
			self.learning_rate = tf.placeholder(tf.float32,name='learning_rate')
			optimizer          = optimizer(self.learning_rate)
        		self.x             = tf.placeholder(tf.float32, shape=input_shape,name='x')
        	        self.y_            = tf.placeholder(tf.int32, shape=[input_shape[0]],name='y')
        	        self.test_phase    = tf.placeholder(tf.bool,name='phase')
        	        self.prediction,self.layers        = model_class.get_layers(self.x,input_shape,test=self.test_phase)
			self.W_=self.layers[1].W
			count_number_of_params()
			self.loss = tf.reduce_mean(tf.pow(self.x-self.prediction,2))#self.crossentropy_loss
        	        self.variables     = tf.trainable_variables()
        	        print "VARIABLES",self.variables
                        self.apply_updates      = opt.apply(self.loss,self.variables)#.values()+tf.get_collection(tf.GraphKeys.UPDATE_OPS))#+updates.values()
		self.session.run(tf.global_variables_initializer())
	def _fit(self,X,update_time=50):
		self.e+=1
        	n_train    = X.shape[0]/self.batch_size
        	train_loss = []
		p = permutation(len(X))
        	for i in xrange(n_train):
			here=p[i*self.batch_size:(i+1)*self.batch_size]
                        self.session.run(self.apply_updates,feed_dict={self.x:X[here],self.test_phase:True,self.learning_rate:float32(self.lr)})#float32(self.lr/sqrt(self.e))})
			if(i%update_time==0):
                                train_loss.append(self.session.run(self.loss,feed_dict={self.x:X[here],self.test_phase:True}))
                        if(i%100 ==0):
                            print i,n_train,train_loss[-1]
        	return train_loss
        def fit(self,X,X_test,n_epochs=5,return_train_accu=0):
		if(n_epochs==0):
			return [0],[0],[]
		train_loss = []
		train_accu = []
		test_loss  = []
		self.e     = 0
		W          = []
                n_test     = X_test.shape[0]/self.batch_size
		for i in xrange(n_epochs):
			print "epoch",i
			train_loss.append(self._fit(X))
			# NOW COMPUTE TEST SET ACCURACY
                	acc1 = 0.0
			W.append(self.session.run(self.W_))
                	for j in xrange(n_test):
                	        acc1+=self.session.run(self.loss,feed_dict={self.x:X_test[self.batch_size*j:self.batch_size*(j+1)],self.test_phase:False})
                	test_loss.append(acc1/n_test)
        	return concatenate(train_loss),test_loss,W
	def predict(self,X):
		n = X.shape[0]/self.batch_size
		preds = []
		for j in xrange(n):
                    preds.append(self.session.run(self.prediction,feed_dict={self.x:X[self.batch_size*j:self.batch_size*(j+1)],self.test_phase:False}))
                return concatenate(preds,axis=0)
        def get_templates(self,X):
                templates = []
                n_batch = X.shape[0]/self.batch_size
                for i in xrange(n_batch):
                        t=self.session.run(self.templates,feed_dict={self.x:X[i*self.batch_size:(i+1)*self.batch_size].astype('float32'),self.test_phase:False})
                        if(len(X.shape)>2):
                                templates.append(transpose(t,[1,0,2,3,4]))
                        else:
                                templates.append(transpose(t,[1,0,2]))
                return concatenate(templates,axis=0)





class DenseCNN:
        def __init__(self,bn=1,n_classes=10,global_beta=1,pool_type='BETA',init_W = tf.contrib.layers.xavier_initializer(uniform=True),init_b = tf.constant_initializer(0.),use_beta=1,nonlinearity=tf.nn.relu,centered=0,ortho=0):
                self.nonlinearity = nonlinearity
                self.bn          = bn
		self.centered=centered
                self.n_classes   = n_classes
		self.global_beta = global_beta
		self.ortho = ortho
		self.pool_type   = pool_type
		self.layers = 0
		self.use_beta    = use_beta
		self.init_W = init_W
		self.init_b = init_b
        def get_layers(self,input_variable,input_shape,test):
		if(self.layers==0):
			extra_layers = []
	                layers = [InputLayer(input_shape,input_variable)]
	                layers.append(Conv2DLayer(layers[-1],32,5,test=test,bn=self.bn,init_W=self.init_W,init_b=self.init_b,first=True,centered=self.centered,ortho=self.ortho))
                        layers.append(NonlinearityLayer(layers[-1],nonlinearity=self.nonlinearity,use_beta=self.use_beta,global_beta=self.global_beta,training=test,bn=self.bn))
	                layers.append(Pool2DLayer(layers[-1],2,pool_type=self.pool_type))
	                layers.append(Conv2DLayer(layers[-1],64,3,test=test,bn=self.bn,init_W=self.init_W,init_b=self.init_b,centered=self.centered,ortho=self.ortho))
                        layers.append(NonlinearityLayer(layers[-1],nonlinearity=self.nonlinearity,use_beta=self.use_beta,global_beta=self.global_beta,training=test,bn=self.bn))
	                layers.append(Pool2DLayer(layers[-1],2,pool_type=self.pool_type))
	                layers.append(Conv2DLayer(layers[-1],128,3,test=test,bn=self.bn,init_W=self.init_W,init_b=self.init_b,centered=self.centered,ortho=self.ortho))
                        layers.append(NonlinearityLayer(layers[-1],nonlinearity=self.nonlinearity,use_beta=self.use_beta,global_beta=self.global_beta,training=test,bn=self.bn))
			layers.append(GlobalPoolLayer(layers[-1]))
                        layers.append(DenseLayer(layers[-1],self.n_classes,training=test,bn=0,init_W=self.init_W,init_b=self.init_b,ortho=0))
			self.layers = layers
			self.extra_layers = layers[1::2]
			global_rec = tf.gradients(layers[-1].output,input_variable,layers[-1].output)[0]
	                return global_rec,self.layers


def inverter(l1,l2,x):
	return tf.gradients(l1.output,l2.output,x)[0]


def upsample(x):
	z=tf.zeros((tf.shape(x)[0],tf.shape(x)[1]*2,tf.shape(x)[2]*2,tf.shape(x)[3]))
	y=z[:,::2,::2,:]+z[:,1::2,1::2,:]
	return tf.gradients(y,z,x)[0]

class DenseCNN2:
        def __init__(self,bn=1,n_classes=10,global_beta=1,pool_type='BETA',init_W = tf.contrib.layers.xavier_initializer(uniform=True),init_b = tf.constant_initializer(0.),use_beta=1,nonlinearity=tf.nn.relu,centered=0,ortho=0):
                self.nonlinearity = nonlinearity
                self.bn          = bn
		self.centered=centered
                self.n_classes   = n_classes
		self.global_beta = global_beta
		self.ortho = ortho
		self.pool_type   = pool_type
		self.layers = 0
		self.use_beta    = use_beta
		self.init_W = init_W
		self.init_b = init_b
        def get_layers(self,input_variable,input_shape,test):
		if(self.layers==0):
			extra_layers = []
	                layers = [InputLayer(input_shape,input_variable)]
	                layers.append(Conv2DLayer(layers[-1],32,5,test=test,bn=self.bn,init_W=self.init_W,init_b=self.init_b,first=True,centered=self.centered,ortho=self.ortho))
                        layers.append(NonlinearityLayer(layers[-1],nonlinearity=self.nonlinearity,use_beta=self.use_beta,global_beta=self.global_beta,training=test,bn=self.bn))
	                layers.append(Pool2DLayer(layers[-1],2,pool_type=self.pool_type))
	                layers.append(Conv2DLayer(layers[-1],64,3,test=test,bn=self.bn,init_W=self.init_W,init_b=self.init_b,centered=self.centered,ortho=self.ortho))
                        layers.append(NonlinearityLayer(layers[-1],nonlinearity=self.nonlinearity,use_beta=self.use_beta,global_beta=self.global_beta,training=test,bn=self.bn))
	                layers.append(Pool2DLayer(layers[-1],2,pool_type=self.pool_type))
	                layers.append(Conv2DLayer(layers[-1],128,3,test=test,bn=self.bn,init_W=self.init_W,init_b=self.init_b,centered=self.centered,ortho=self.ortho))
                        layers.append(NonlinearityLayer(layers[-1],nonlinearity=self.nonlinearity,use_beta=self.use_beta,global_beta=self.global_beta,training=test,bn=self.bn))
			layers.append(GlobalPoolLayer(layers[-1]))
                        layers.append(DenseLayer(layers[-1],self.n_classes,training=test,bn=0,init_W=self.init_W,init_b=self.init_b,ortho=0))
			a1 = inverter(layers[-1],layers[-2],layers[-1].output)
			a2 = inverter(layers[-2],layers[-3],a1)
			a3 = tf.nn.relu(a2)
			a4 = inverter(layers[-4],layers[-5],a3)
			a5 = upsample(a4)
			a6 = tf.nn.relu(a5)
                        a7 = inverter(layers[-7],layers[-8],a6)
			a8 = upsample(a7)
                        a9 = tf.nn.relu(a8)
                        a10 = inverter(layers[-10],layers[-11],a9)
	                return a10,layers




