from torchvision import models
import torch
from torchvision import transforms
from PIL import Image
import os

def Netowork_test(test):
    transform = transforms.Compose([    # [1]
        transforms.Resize(256),         # [2]
        transforms.CenterCrop(224),     # [3]
        transforms.ToTensor(),          # [4]
        transforms.Normalize(           # [5]
            mean=[0.485, 0.456, 0.406], # [6]
            std=[0.229, 0.224, 0.225]   # [7]
        )])
    img = Image.open(test)
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)
    Alexnet = models.alexnet(True)
    Alexnet.eval()
    out_alexnet= Alexnet(batch_t)
    VGG_16 = models.vgg16_bn(True)
    VGG_16.eval()
    out_Vgg_16 = VGG_16(batch_t)
    Googlenet = models.googlenet(True)
    Googlenet.eval()
    out_googlenet = Googlenet(batch_t)
    # Inception = models.inception_v3(True)
    # Inception.eval()
    # out_inception = Inception(batch_t)

    test_app = "\n" + test + "\n"

    with open('imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]
    print("Alex-Net")
    _, indices = torch.sort(out_alexnet, descending=True)
    percentage = torch.nn.functional.softmax(out_alexnet, dim=1)[0] * 100
    # [print (classes[idx], percentage[idx].item()) for idx in indices[0][:5]]
    file1 = open("Alexnet_without_prob.txt", "a+")
    file2 = open("Alexnet.txt", "a+")
    file1.write(test_app)
    file2.write(test_app)

    for idx in indices[0][:5]:
        x, y = classes[idx], percentage[idx].item()
        file1.writelines("__" + str(x))
        file2.writelines(str(x) + str(y) + "\n")
    file1.close()
    file2.close()

    print("VGG-16")
    _, indices = torch.sort(out_Vgg_16, descending=True)
    percentage = torch.nn.functional.softmax(out_Vgg_16, dim=1)[0] * 100
    # [print(classes[idx], percentage[idx].item()) for idx in indices[0][:5]]
    file1 = open("VGG_16_without_prob.txt", "a+")
    file2 = open("VGG_16.txt", "a+")
    file1.write(test_app)
    file2.write(test_app)
    for idx in indices[0][:5]:
        x, y = classes[idx], percentage[idx].item()
        file1.writelines("__" + str(x))
        file2.writelines(str(x) + str(y) + "\n")
    file1.close()
    file2.close()

    print("Googlenet")
    _, indices = torch.sort(out_googlenet, descending=True)
    percentage = torch.nn.functional.softmax(out_googlenet, dim=1)[0] * 100
    # [print(classes[idx], percentage[idx].item()) for idx in indices[0][:5]]

    file1 = open("GoogleNet_without_prob.txt", "a+")
    file2 = open("GoogleNet.txt", "a+")
    file1.write(test_app)
    file2.write(test_app)
    for idx in indices[0][:5]:
        x, y = classes[idx], percentage[idx].item()
        file1.writelines("__" + str(x))
        file2.writelines(str(x) + str(y) + "\n")
    file1.close()
    file2.close()
    #
    # print("Inception")
    # _, indices = torch.sort(out_inception, descending=True)
    # percentage = torch.nn.functional.softmax(out_inception, dim=1)[0] * 100
    # # [print(classes[idx], percentage[idx].item()) for idx in indices[0][:5]]
    # file1 = open("Inception_without_prob.txt", "a+")
    # file2 = open("Inception.txt", "a+")
    # file1.write(test_app)
    # file2.write(test_app)
    # for idx in indices[0][:5]:
    #     x, y = classes[idx], percentage[idx].item()
    #     file1.writelines("__" + str(x))
    #     file2.writelines(str(x) + str(y) + "\n")
    # file1.close()
    # file2.close()

directory = r'Mini_dataset'
for filename in os.listdir(directory):
  if filename.endswith(".JPEG"):
    test = os.path.join(directory,filename)
    print(test)
    Netowork_test(test)
