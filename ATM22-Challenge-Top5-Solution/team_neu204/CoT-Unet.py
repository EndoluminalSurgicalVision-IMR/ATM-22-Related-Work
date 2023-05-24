import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os








class CoTAttention(nn.Module):

    def __init__(self, dim=512,kernel_size=3,res=False):
        super().__init__()
        self.res = res
        self.dim=dim
        self.kernel_size=kernel_size

        self.key_embed=nn.Sequential(
            nn.Conv3d(dim,dim,kernel_size=kernel_size,padding=kernel_size//2,groups=4,bias=False),
            nn.BatchNorm3d(dim),
            nn.ReLU()
        )
        self.value_embed=nn.Sequential(
            nn.Conv3d(dim,dim,1,bias=False),
            nn.BatchNorm3d(dim)
        )

        factor=4
        self.attention_embed=nn.Sequential(
            nn.Conv3d(2*dim,2*dim//factor,1,bias=False),
            nn.BatchNorm3d(2*dim//factor),
            nn.ReLU(),
            nn.Conv3d(2*dim//factor,kernel_size*kernel_size*dim,1)
        )


    def forward(self, x):
        bs,c,h,w,d=x.shape
        k1=self.key_embed(x) #bs,c,h,w
        v=self.value_embed(x).view(bs,c,-1) #bs,c,h,w

        y=torch.cat([k1,x],dim=1) #bs,2c,h,w
        att=self.attention_embed(y) #bs,c*k*k,h,w
        att=att.reshape(bs,c,self.kernel_size*self.kernel_size,h,w,d)
        att=att.mean(2,keepdim=False).view(bs,c,-1) #bs,c,h*w
        k2=F.softmax(att,dim=-1)*v
        k2=k2.view(bs,c,h,w,d)

        out = k1+k2


        return out
class UNet3D(nn.Module):
	def __init__(self,in_channels=1, out_channels=1
       ,Dmax=128, Hmax=128, Wmax=128):
		super(UNet3D, self).__init__()


		self._in_channels = in_channels
		self._out_channels = out_channels
		self.upsampling4 = nn.Upsample(scale_factor=4)
		self.upsampling8 = nn.Upsample(scale_factor=8)
		self.pooling = nn.MaxPool3d(kernel_size=(2, 2, 2))
		self.conv1 = nn.Sequential(
			nn.Conv3d(in_channels=self._in_channels, out_channels=8, kernel_size=3, stride=1, padding=1),
			nn.InstanceNorm3d(8),
			nn.ReLU(inplace=True),
			nn.Conv3d(8, 16, 3, 1, 1),
			nn.InstanceNorm3d(16),
			nn.ReLU(inplace=True))
		# self.conv1x1 = nn.Sequential(
		# 	nn.Conv3d(in_channels=self._in_channels, out_channels=16, kernel_size=1, stride=1),
		# 	nn.InstanceNorm3d(16),
		# 	nn.ReLU(inplace=True),)
		self.conv2 = nn.Sequential(
			CoTAttention(16,3),
			nn.InstanceNorm3d(16),
			nn.ReLU(inplace=True),
			nn.Conv3d(16, 32, 3, 1, 1),
			nn.InstanceNorm3d(32),
			nn.ReLU(inplace=True))
		# self.conv2x1 = nn.Sequential(
		# 	nn.Conv3d(in_channels=16, out_channels=32, kernel_size=1, stride=1),
		# 	nn.InstanceNorm3d(32),
		# 	nn.ReLU(inplace=True), )

		self.conv3 = nn.Sequential(
			CoTAttention(32,3),
			nn.InstanceNorm3d(32),
			nn.ReLU(inplace=True),
			nn.Conv3d(32, 64, 3, 1, 1),
			nn.InstanceNorm3d(64),
			nn.ReLU(inplace=True))
		# self.conv3x1 = nn.Sequential(
		# 	nn.Conv3d(in_channels=32, out_channels=64, kernel_size=1, stride=1),
		# 	nn.InstanceNorm3d(64),
		# 	nn.ReLU(inplace=True), )
	
		self.conv4 = nn.Sequential(
			CoTAttention(64,3),
			nn.InstanceNorm3d(64),
			nn.ReLU(inplace=True),
			nn.Conv3d(64, 128, 3, 1, 1),
			nn.InstanceNorm3d(128),
			nn.ReLU(inplace=True))
		# self.conv4x1 = nn.Sequential(
		# 	nn.Conv3d(in_channels=64, out_channels=128, kernel_size=1, stride=1),
		# 	nn.InstanceNorm3d(128),
		# 	nn.ReLU(inplace=True), )

		self.conv5 = nn.Sequential(
			CoTAttention(128, 3),
			nn.InstanceNorm3d(128),
			nn.ReLU(inplace=True),
			nn.Conv3d(128, 256, 3, 1, 1),
			nn.InstanceNorm3d(256),
			nn.ReLU(inplace=True))
		# self.conv5x1 = nn.Sequential(
		# 	nn.Conv3d(in_channels=128, out_channels=256, kernel_size=1, stride=1),
		# 	nn.InstanceNorm3d(256),
		# 	nn.ReLU(inplace=True), )

		self.conv6 = nn.Sequential(
			nn.Conv3d(256 + 128, 128, kernel_size=3, stride=1, padding=1),
			nn.InstanceNorm3d(128),
			nn.ReLU(inplace=True),
			CoTAttention(128,3),
			nn.InstanceNorm3d(128),
			nn.ReLU(inplace=True))
		# self.conv6x1 = nn.Sequential(
		# 	nn.Conv3d(in_channels=384, out_channels=128, kernel_size=1, stride=1),
		# 	nn.InstanceNorm3d(128),
		# 	nn.ReLU(inplace=True), )

		self.conv7 = nn.Sequential(
			nn.Conv3d(64 + 128, 64, 3, 1, 1),
			nn.InstanceNorm3d(64),
			nn.ReLU(inplace=True),
			CoTAttention(64,3),
			nn.InstanceNorm3d(64),
			nn.ReLU(inplace=True))
		# self.conv7x1 = nn.Sequential(
		# 	nn.Conv3d(in_channels=128 + 64, out_channels=64, kernel_size=1, stride=1),
		# 	nn.InstanceNorm3d(64),
		# 	nn.ReLU(inplace=True), )

		self.conv8 = nn.Sequential(
			nn.Conv3d(32 + 64, 32, 3, 1, 1),
			nn.InstanceNorm3d(32),
			nn.ReLU(inplace=True),
			CoTAttention(32,3),
			nn.InstanceNorm3d(32),
			nn.ReLU(inplace=True))
		# self.conv8x1 = nn.Sequential(
		# 	nn.Conv3d(in_channels=64 + 32, out_channels=32, kernel_size=1, stride=1),
		# 	nn.InstanceNorm3d(32),
		# 	nn.ReLU(inplace=True), )
		
		if self._coord:
			num_channel_coord = 3
		else:
			num_channel_coord = 0
		self.conv9 = nn.Sequential(
			nn.Conv3d(16 + 32 + num_channel_coord, 16, 3, 1, 1),
			nn.InstanceNorm3d(16),
			nn.ReLU(inplace=True),
			CoTAttention(16,3),
			nn.InstanceNorm3d(16),
			nn.ReLU(inplace=True))
		# self.conv9x1 = nn.Sequential(
		# 	nn.Conv3d(in_channels=32 + 16, out_channels=16, kernel_size=1, stride=1),
		# 	nn.InstanceNorm3d(16),
		# 	nn.ReLU(inplace=True), )
	
		self.sigmoid = nn.Sigmoid()
		self.conv10 = nn.Conv3d(16, self._out_channels, 1, 1, 0)

	def forward(self, input):

		conv1 = self.conv1(input)


		x = self.pooling(conv1)

		conv2 = self.conv2(x)
		# res2   = self.conv2x1(x)
		# conv2 = conv2 + res2
		x = self.pooling(conv2)

		conv3 = self.conv3(x)
		# res3 = self.conv3x1(x)
		# conv3 = conv3 + res3
		x = self.pooling(conv3)

		conv4 = self.conv4(x)
		# res4 = self.conv4x1(x)
		# conv4 = conv4 + res4
		x = self.pooling(conv4)


		conv5 = self.conv5 (x)
		# res5 = self.conv5x1(x)
		# conv5 = conv5 + res5



		x = self.upsampling(conv5)
		x = torch.cat([x, conv4], dim=1)
		conv6 = self.conv6(x)





		x = self.upsampling(conv6)
		x = torch.cat([x, conv3], dim=1)
		conv7 = self.conv7(x)


		x = self.upsampling(conv7)
		x = torch.cat([x, conv2], dim=1)
		conv8 = self.conv8(x)





		x = self.upsampling(conv8)


		x = torch.cat([x, conv1], dim=1)
		conv9 = self.conv9(x)




		x = self.conv10(conv9)
		x = self.sigmoid(x)



		return x









