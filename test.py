import csv
import cv2
import os
from PIL import Image
from model import *
import torch
import numpy as np
import torchvision.transforms as transforms

csvFile = open("res.csv","w",newline='')  # 创建csv文件
writer = csv.writer(csvFile)
# 先写入columns_name
writer.writerow(["image_name", "predict",'个人资料','申请资料','检查报告','费用清单','诊断书/证明','治疗记录'])  # 写入列的名称

# 读入图片
# test_root = './additional/'
test_root = 'Y:/taibao/data/val_set/'
img_test = os.listdir(test_root)

#creat model
model = PMG(512, 6)
model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(('./pth/fold1_best_acc.pth'),map_location=torch.device('cpu')).items()})
model.eval()

#preprocess imgae
MEAN = [0.79669344,0.7912885,0.76414025]
STD = [0.21069272,0.21318783,0.21557616]
transform1 = transforms.Compose([
			transforms.Resize((512, 512)),
			transforms.ToTensor(),
			transforms.Normalize(tuple(MEAN), tuple(STD))
    ])

for i in range(len(img_test)):
	if(img_test[i].endswith('jpg')): #文件夹下会有readme等其它文件
		rd_img = cv2.imread(test_root + img_test[i])
		img = Image.open(test_root + img_test[i])
		print(img_test[i])
		img = img.convert('RGB')

		input = transform1(img) # 这里经过转换后输出的input格式是[C,H,W],网络输入还需要增加一维批量大小B，
		input = input.unsqueeze(0) #增加一维，输出的img格式为[1,C,H,W]

		# print(input.size())#
		output_1, output_2, output_3, output_concat = model(input)
		score = output_1 + output_2 + output_3 + output_concat  # 将图片输入网络得到输出
		probability = torch.nn.functional.softmax(score, dim=1)  # 计算softmax，即该图片属于各类的概率
		max_value, index = torch.max(probability, 1)  # 找到最大概率对应的索引号，该图片即为该索引号对应的类别
		class_index = index.item()
		print(class_index)
		#'个人资料','申请资料','检查报告','费用清单','诊断书/证明','治疗记录'
		if class_index ==0:
			class_index ='个人资料'
		elif class_index == 1:
			class_index ='申请资料'
		elif class_index == 2:
			class_index = '检查报告'
		elif class_index == 3:
			class_index ='费用清单'
		elif class_index == 4:
			class_index = '诊断书/证明'
		else:
			class_index = '治疗记录'
		probability = np.round(probability.cpu().detach().numpy(), 3)
		writer.writerow(
			[img_test[i],class_index, probability[0][0], probability[0][1], probability[0][2], probability[0][3],
			 probability[0][4],probability[0][5]])
csvFile.close()