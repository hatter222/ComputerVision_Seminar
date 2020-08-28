import PIL
import cv2
import numpy as np
import torch
from torchvision import models
#print(dir(models))
from torchvision import transforms
from PIL import Image,ImageOps
import os
import time
from datetime import timedelta
def resnet_test(test):
  transform = transforms.Compose([            #[1]
  transforms.Resize(256),                    #[2]
  transforms.CenterCrop(224),                #[3]
  transforms.ToTensor(),                     #[4]
  transforms.Normalize(                      #[5]
  mean=[0.485, 0.456, 0.406],                #[6]
  std=[0.229, 0.224, 0.225]                  #[7]
  )])

  lowervalue=10
  uppervalue=250
  treshold=0.05
  img = Image.open(test)
  #img = np.array(img)
  # random_matrix = np.random.random(img.shape)
  # img[random_matrix >= (1 - treshold)] = uppervalue
  # img[random_matrix <= treshold] = lowervalue
  # f_img = Image.fromarray(img)

 # f_img = ImageOps.invert(img)
  img_t = transform(img)
  #f_img.save('transpose-output.png')
  batch_t = torch.unsqueeze(img_t, 0)
  resnet_34 = models.resnet34(pretrained=True)
  resnet_34.eval()
  out_34 = resnet_34(batch_t)

  resnet_50 = models.resnet50(pretrained=True)
  resnet_50.eval()
  out_50 = resnet_50(batch_t)

  resnet_101 = models.resnet101(pretrained=True)
  resnet_101.eval()
  out_101 = resnet_101(batch_t)

  resnet_152 = models.resnet152(pretrained=True)
  resnet_152.eval()
  out_152 = resnet_152(batch_t)
  test_app = "\n"+test+"\n"

  with open('imagenet_classes.txt') as f:
   classes = [line.strip() for line in f.readlines()]
  print("Resnet - 34")
  _, indices = torch.sort(out_34, descending=True)
  percentage = torch.nn.functional.softmax(out_34, dim=1)[0] * 100
  #[print (classes[idx], percentage[idx].item()) for idx in indices[0][:5]]
  file1 = open("results_34_without_prob.txt", "a+")
  file2 = open("results_34.txt","a+")
  file1.write(test_app)
  file2.write(test_app)

  for idx in indices[0][:5]:
   x,y = classes[idx], percentage[idx].item()
   file1.writelines("__"+str(x))
   file2.writelines(str(x) + str(y) + "\n")
  file1.close()
  file2.close()

  print("Resnet - 50")
  _, indices = torch.sort(out_50, descending=True)
  percentage = torch.nn.functional.softmax(out_50, dim=1)[0] * 100
  #[print(classes[idx], percentage[idx].item()) for idx in indices[0][:5]]
  file1 = open("results_50_without_prob.txt", "a+")
  file2 = open("results_50.txt", "a+")
  file1.write(test_app)
  file2.write(test_app)
  for idx in indices[0][:5]:
   x,y = classes[idx], percentage[idx].item()
   file1.writelines("__"+str(x))
   file2.writelines(str(x)+ str(y)+ "\n")
  file1.close()
  file2.close()

  print("Resnet - 101")
  _, indices = torch.sort(out_101, descending=True)
  percentage = torch.nn.functional.softmax(out_101, dim=1)[0] * 100
  #[print(classes[idx], percentage[idx].item()) for idx in indices[0][:5]]

  file1 = open("results_101_without_prob.txt", "a+")
  file2 = open("results_101.txt", "a+")
  file1.write(test_app)
  file2.write(test_app)
  for idx in indices[0][:5]:
   x,y = classes[idx], percentage[idx].item()
   file1.writelines("__"+str(x))
   file2.writelines(str(x) + str(y) + "\n")
  file1.close()
  file2.close()


  print("Resnet - 152")
  _, indices = torch.sort(out_152, descending=True)
  percentage = torch.nn.functional.softmax(out_152, dim=1)[0] * 100
  #[print(classes[idx], percentage[idx].item()) for idx in indices[0][:5]]
  file1 = open("results_152_without_prob.txt", "a+")
  file2 = open("results_152.txt", "a+")
  file1.write(test_app)
  file2.write(test_app)
  for idx in indices[0][:5]:
   x,y = classes[idx], percentage[idx].item()
   file1.writelines("__"+str(x))
   file2.writelines(str(x) + str(y) + "\n")
  file1.close()
  file2.close()


start= time.monotonic()
directory = r'Mini_dataset'
for filename in os.listdir(directory):
  if filename.endswith(".JPEG"):
    test = os.path.join(directory,filename)
    print(test)
    try:
      resnet_test(test)
    except:
      print("error occured")
    finally:
      print("all done")
end=time.monotonic()
print(timedelta(seconds=end-start))