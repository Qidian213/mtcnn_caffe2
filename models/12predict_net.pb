
PNetF
data
conv1_w
conv1_bconv1"Conv*

stride*
pad *

kernel"
conv1
conv1_Slopeconv1"PReluG
conv1pool1"MaxPool*

stride*
pad *

kernel*
order"NCHWG
pool1
conv2_w
conv2_bconv2"Conv*

stride*
pad *

kernel"
conv2
conv2_Slopeconv2"PReluG
conv2
conv3_w
conv3_bconv3"Conv*

stride*
pad *

kernel"
conv3
conv3_Slopeconv3"PReluM
conv3
	conv4-1_w
	conv4-1_bconv4-1"Conv*

stride*
pad *

kernelM
conv3
	conv4-2_w
	conv4-2_bconv4-2"Conv*

stride*
pad *

kernel
conv4-1prob1"Softmax:data:conv1_w:conv1_b:conv1_Slope:conv2_w:conv2_b:conv2_Slope:conv3_w:conv3_b:conv3_Slope:	conv4-1_w:	conv4-1_b:	conv4-2_w:	conv4-2_bBprob1Bconv4-2