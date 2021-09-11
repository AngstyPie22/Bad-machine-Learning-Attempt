import numpy as np
import math as math
import random as ran #used to create random values for the weights and biases when first creating the network.
import matplotlib.pyplot as plt #used to plot the error rate to give a visual aid in checking that the network is functioning.
import DataSet_3x3_Squares_Extended_label as DSQ3x3 # this imports the data set from the projects folder
dta = DSQ3x3.Data[0][0] #a sample of the inputs we will be using to put into the network. (used to create the input layer size)
exp = DSQ3x3.Data[0][1]	#a sample of the expected outputs(or classification label) for the given data sample (used to create the output layer size)
class KpNet_V2:
	weight_list = [] #Declaring the class variables to be used between the methods
	bias_list = []
	layer_numbers = []	#the number of neurons in the hidden layers
	learning_rate = None
	epocs = None		
	activ = None		#Activaiton funciton
	data = None			#input data
	expected = None		#data labels
	output_list = []	#keeps track of the ouputs for use in backpropigation
	Error_values = []	#for use when creating the error graph in start()
	def __init__(self,Data,Expected,Hidden_sizes,Activation,Learning_rate,Epocs):	#sets the class variables to the values set when creating in instance of the network
		Hidden_sizes.insert(0,len(Data))	#these two lines set the number of neurons in the input and output layers based on the # of variables in the data/expected values
		Hidden_sizes.append(len(Expected))	#<--/
		print(Hidden_sizes)
		self.layer_numbers = Hidden_sizes
		self.learning_rate = Learning_rate
		self.epocs = Epocs
		self.activ = Activation
		self.data = Data
		self.output_list.append(np.array([Data])) #adds the current input value to the first slot in the output layer for use in backpropigation
		self.expected = Expected
	def weights_and_biases(self):	#creates the weights and biases based on the input, output, and hidden layer values set in the layer_numbers variable and adds them to the class variables weight/bias_list
		for x in range(len(self.layer_numbers)):
			self.weight_list.append(np.random.randn(self.layer_numbers[x-1],self.layer_numbers[x]))
			self.bias_list.append(np.random.randn(1,self.layer_numbers[x]))

	def activation(self,function,x,deriv):	#applies the various supported activation funcitons to x (the outputs of each layer). the specific function is set by the user when creating the network except for the softmax on the output layer.
		#x = np.clip(x,-500,500)
		if function.lower() == "relu":
			if deriv:
				x[x<=0] = 0
				x[x>0] = 1
				return x
			else:
				return x*(x>0)
		elif function.lower() in ['sig','sigmoid']:
			if deriv:
				return x * (1-x)
			else:
				return 1/(1 + np.exp(-x))
		elif function.lower() in ['softmax','soft']:
			if deriv:
				return np.exp(x)/(np.sum(np.exp(x)))
			else:
				exp_val = np.exp(x-np.max(x,axis=1,keepdims=True))
				return exp_val/np.sum(exp_val, axis=1,keepdims=True)
		else:
			print('Activation Module Error')

	def Forward_Back(self):	#preforms the forward(Lines 62-69) and backpropigation(Lines 70-79) for the network
		output = self.data
		for i in range(len(self.layer_numbers)-1):
			WeightMath = np.dot(output,self.weight_list[i])+self.bias_list[i]
			output = self.activation(self.activ,WeightMath,False)
			self.output_list.append(output)
		i = len(self.layer_numbers)-1
		WeightMath = np.dot(output,self.weight_list[i])+self.bias_list[i]
		output = self.activation('soft',WeightMath,False)
		print(self.data)
#--------------------------Backpropigation-------------------------------
		Error = .5*np.square(self.expected - self.activation('soft',output,True))
		# New Error Back.Error = -np.log(Create.OList[len(Create.OList)-1][0][Expected])
		Delta = ((Error*self.learning_rate) * output)
		self.weight_list[len(self.weight_list)-1] += self.weight_list[len(self.weight_list)-1].dot(Delta.T)
		self.bias_list[len(self.bias_list)-1] += Delta
		for i in reversed(range(len(self.weight_list))):
			Delta = ((Delta.dot(self.weight_list[i].T))*self.activation(self.activ,self.output_list[i],True))*self.learning_rate
			self.weight_list[i-1] += (self.output_list[i-1].T.dot(Delta))
			self.bias_list[i-1] += Delta
		self.Error_values.append(np.sum(Error)/9)
	def start(self): #initiates the weights/bias creation method and the forward/backpropigation method. 
		self.weights_and_biases()
		for i in range(self.epocs):		#Uses the # of epocs to set how many times the data will be put through.
			i = ran.randrange(len(DSQ3x3.Data)) #sets a random index within the total number of samples in the data
			self.Forward_Back()
		XValues=range(0,self.epocs)
		plt.plot(XValues,self.Error_values) #plots the sum of the error (Y) per epoc (X)
		plt.show()

new = KpNet_V2(dta,exp,[9,9],'sig',.01,30) #creating an instance of the network and setting the various settings (|input|,|expected output|,|hidden layer numbers|,|activation funciton|,|learning rate|,|#of epocs|)
new.start() #calling the start method to initiate the networks weights/biases and pushing the data through them.
