from pylab import *
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import cPickle

execfile('utils.py')
execfile('models.py')
execfile('lasagne_tf.py')

DATASET    = sys.argv[-2]
bn         = 0#int(sys.argv[-2])
learn_beta = 1



start = int(sys.argv[-3])

ne=100

for nonlin in ['relu']:
#	model = models[int(sys.argv[-1])]
#	model_name = models_names[int(sys.argv[-1])]#for model,model_name in zip([DenseCNN,largeCNN],['SmallCNN','LargeCNN']):
#	data = dict()
	for k in xrange(start,start+5):
		if(1):#for bn in [1,0]:#if(1):#for use_beta in [0,1]:
			if(int(sys.argv[-1])==0):
				lrs= [0.005,0.001,0.0002]
			else:
                                lrs= [0.001,0.0002]
                        for lr in lrs:
		                name   = 'new'+DATASET
				x_train,x_test,y_train,y_test,c,n_epochs,input_shape=load_utility(DATASET)
####
				m = DenseCNN(bn=bn,global_beta=0,pool_type='MAX',nonlinearity=nonlin,centered=0,ortho=0,n_classes=c)
				model1    = DNNClassifier(input_shape,m,optimizer = tf.train.AdamOptimizer,lr=lr,learn_beta=learn_beta)
				train_loss,test_accu,W = model1.fit(x_train,x_test,n_epochs=ne,return_train_accu=1)
                                data0=[train_loss,test_accu,W]
####
                                m = DenseCNN2(bn=bn,global_beta=0,pool_type='MAX',nonlinearity=nonlin,centered=0,ortho=0,n_classes=c)
                                model1    = DNNClassifier(input_shape,m,optimizer = tf.train.AdamOptimizer,lr=lr,learn_beta=learn_beta)
                                train_loss,test_accu,W = model1.fit(x_train,x_test,n_epochs=ne,return_train_accu=1)
                                data1=[train_loss,test_accu,W]
####
			        f = open('./'+name+str(k)+'.pkl','wb')
			        cPickle.dump([data0,data1],f)
			        f.close()
		
	
	


