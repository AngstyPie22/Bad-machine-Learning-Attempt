import numpy as np
import math as math
import random as ran #used to create random values for the weights and biases when first creating the network.
import matplotlib.pyplot as plt #used to plot the error rate to give a visual aid in checking the network is functioning.
import DataSet_3x3_Squares_Extended_label as DSQ3x3 # this imports the data set from the projects folder
class KPNet():
	def __init__(self,Input_Data,Epocs,Learning_Rate,Hidden_Sizes,Activation):
		self.Epocs = Epocs
		self.Learning_Rate = Learning_Rate
		self.Hidden_Sizes = Hidden_Sizes
		self.Activation = Activation
		self.Input_Data = Input_Data
		self.W = []
		self.B = []
		def Weights(me_size,Inputs_size):#Creates weights for all layers
			Weights = np.random.randn(Inputs_size,me_size)
			return Weights
		def Biases(me_size):#sets up bias values for layers
			Biases = np.random.randn(1,me_size)
			return Biases
		def Activation(function,x,deriv):#changes what activation function has been set for use by the user
			if function.lower() == "relu":
				if deriv == False:
					x = x*(x>0)
					return x
				elif deriv ==True:
					x[x<=0] = 0
					x[x>0] = 1
					return x
			elif function.lower() in ['sig','sigmoid']:
				if deriv == False:
					x = 1./(1 + np.exp(x))
					return x
				elif deriv == True:
					x = x * (1-x)
					return x
			elif function.lower() in ['softmax','soft']:
				if deriv == False:
					exp_val = np.exp(x-np.max(x,axis=1,keepdims=True))
					x = exp_val/np.sum(exp_val, axis=1,keepdims=True)
					return x
				if deriv == True:
					return np.exp(x)/(np.sum(np.exp(x)))
			else:
				print('Activation Module Error')
				return x
		def Create():#creates empty lists to store the bias, weight, and output matrices.
			Create.WList = []
			Create.BList = []
			Create.OList = []
			l_num = 0
			for i in range(Go.MasLen-1):
				Create.WList.append(Weights(Go.Master_Sizes[l_num+1],Go.Master_Sizes[l_num]))
				Create.BList.append(Biases(Go.Master_Sizes[l_num+1]))
				l_num += 1 
		def Forward(Data_Sample,W,B,Activ):#takes the input and pushes it through the layers using weights, biases and activation function
			Create.OList = []
			Create.OList.append(np.array([Data_Sample]))
			arps = 0
			for i in range(Go.MasLen-2):
				Wm = np.dot(Create.OList[arps],W[arps])+B[arps]
				RL = Activation(Activ,Wm,False)
				Create.OList.append(RL)
				arps += 1
			Out_Wm = np.dot(Create.OList[len(Create.OList)-1],W[len(W)-1])+Create.BList[len(Create.BList)-1]
			Out_fin = Activation('soft',Out_Wm,False)
			Create.OList.append(Out_fin)
		def Back(Expected,W,B,Data,Activ,LR):#takes the output and compares it to the desired output. Determines the error and adjusts the network to lower it.
			Back.WList = np.array([])
			Back.BList = np.array([])
			Back.Out = Create.OList[len(Create.OList)-1]
			#OLD ERROR Back.Error = Expected - Create.OList[len(Create.OList)-1]
			Back.Error = -np.log(Create.OList[len(Create.OList)-1][0][Expected])
			#Back.Overall = float(Back.Error.sum())/len(Back.Error)
			Delta = (Back.Error * Create.OList[len(Create.OList)-1])
			W[len(W)-1] -= (Create.OList[len(Create.OList)-2].T.dot(Delta))*LR
			B[len(B)-1] -= Delta*LR
			arps = len(W)-1
			for i in range(arps):
				Error = Delta.dot(W[arps].T)
				Delta = Error*Activation(Activ,Create.OList[arps],True)
				W[arps-1] -= (Create.OList[arps-1].T.dot(Delta))*LR
				B[arps-1] -= Delta*LR
				Back.WList = W
				Back.BList = B
				arps -= 1
		def Go(Input,Epocs,LearningRate,HiddenSizes,Activation):#functions as a master control where all funcitons are run in correct sequence.
			Go.Master_Sizes = []
			Go.Master_Sizes.append(len(Input[0][0])) #input size
			Go.Master_Sizes.extend(HiddenSizes)
			Go.Master_Sizes.append(len(Input[0][0])) #output size
			Go.MasLen = len(Go.Master_Sizes)
			Error_values = []
			XValues = []
			actual = []
			Epoc = 1
			Create()
			Wl = Create.WList
			Bl = Create.BList
			for i in range(Epocs):
				arpsd = 0
				for i in range(len(Input)):
					Index = ran.randrange(len(Input))
					Input_Data = Input[Index][0]
					Expected_out = Input[Index][0]
					Forward(Input_Data,Wl,Bl,Activation)
					Back(Expected_out,Wl,Bl,Input_Data,Activation,LearningRate)
					Wl = Back.WList
					Bl = Back.BList
					if arpsd >= len(Input)-1:
						Error_values.append(Back.Error)
						XValues.append(Epoc)
					arpsd+=1
				Wl = Back.WList
				Bl = Back.BList
				print('This is Epoc',Epoc)
				#print(Create.OList[len(Create.OList)-1])
				Epoc += 1
			self.W = Wl
			self.B = Bl
			plt.plot(XValues,Error_values)
			plt.show()
		#Go(DSQ3x3.Data,Epocs,Learning_Rate,Hidden_Sizes,'sig')#This is the master control. Use this to set the activation function and data input.
		Go(self.Input_Data,self.Epocs,self.Learning_Rate,self.Hidden_Sizes,self.Activation)
#Net = KPNet(DSQ3x3.Data,200,.1,[20,20,20],'sig')
#Epocs = how many times we put the whole data set through the network.
#Learning_Rate = how much we want to change the network per wrong answer
#Hidden_Sizes = How many layers are in the network and the # of neurons in each layer. Goes from after input to just before output'''
#input and output neurons automatically scale with the number of values in the output and input so you dont need to set them.
