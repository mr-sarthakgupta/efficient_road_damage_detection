from sched import scheduler
from statistics import mean
from turtle import shape
import numpy as np
import torch
import os
from fast_seg import get_colorized_seg_map, get_fastseg_model
from get_segformer import get_segformer, get_segformer_feature_extractor
# from BiSeNet.tools.get_bisenet import get_bisenet
from torch.utils.tensorboard import SummaryWriter
import logging
import argparse
import time
from tqdm import tqdm
from fastseg.image import colorize, blend
from get_dataloader import get_only_nsv_360_images_dataloaders
import torchvision
import torchmetrics
from PIL import Image 

writer = SummaryWriter('tensorboard')
device_1 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device_2 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
segformer = get_segformer().to(device_1)
segformerextractor = get_segformer_feature_extractor()
segformer.eval()

def teacher(inputs):
    im_list = []
    im_list.append(inputs[0])
    im_list.append(inputs[1])
    inputs = segformerextractor(images = im_list, return_tensors="pt")['pixel_values'].to(device_1) 
    return segformer(inputs)

def get_two_classes(vector):
    total = torch.cat((torch.sum(vector, dim = 1, keepdim = True) - torch.unsqueeze(vector[:, 7, :, :], dim = 1), torch.unsqueeze(vector[:, 7, :, :], dim = 1)), dim = 1)
    arg = torch.argmax(vector, dim = 1, keepdim = True)
    arg[arg != 0] = 1
    arg[arg == 0] = 0
    return total, arg

def get_hard_labels(vector):
    vector1 = torch.argmax(vector, dim = 1, keepdim = True)
    vector1 = torch.permute(torch.squeeze(torch.nn.functional.one_hot(vector1, num_classes = 2)), (0, 3, 1, 2))
    return vector1

def train_fn(student, ds_train, i):
    resizer_1 = torchvision.transforms.Resize(size = (1024, 1024))
    resizer_2 = torchvision.transforms.Resize(size = (256, 256))
    sigmoid = torch.nn.Sigmoid()
    student.to(device_2)
    student.train() 
    optimizer = torch.optim.Adam(student.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.4, last_epoch=- 1, verbose=True)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction = 'mean').to(device_2)
    jaccard = torchmetrics.JaccardIndex(19).to(device_2)
    k = 0
    for imgs in ds_train:
        imgs_2 = imgs.to(device_2)
        imgs_1 = resizer_1(imgs)
        imgs_2 = resizer_1(imgs_2)
        teacher_label, teacher_iou = get_two_classes(teacher(imgs_1)['logits'].to(device_2))
        teacher_label = get_hard_labels(teacher_label)
        student_label, student_iou = get_two_classes(student(imgs_2).to(device_2))
        del imgs_1
        del imgs_2
        student_label = resizer_2(student_label)
        student_iou = resizer_2(student_iou)
        # loss = loss_fn(student_label, sigmoid(teacher_label.detach()))
        loss = loss_fn(student_label, teacher_label.detach().float())
        del teacher_label
        # show_label = teacher_iou
        # show_label_2 = student_iou
        iou = jaccard(teacher_iou, student_iou)
        del student_iou
        del teacher_iou
        print(f'train_loss : {loss.item()} | train_IOU : {iou.item()}')
        loss.backward()
        optimizer.step
        optimizer.zero_grad()
        del student_label
        # show_label = torch.squeeze(show_label[0], dim = 0).detach().cpu().numpy()
        # colorized = colorize(show_label)
        # og_img = (255*imgs[0]).permute((1, 2, 0)).type(torch.uint8).detach().cpu().numpy()
        # og_img = Image.fromarray(og_img)
        # og_img.save('og_img.png')
        # colorized.save('tester.png')
        # show_label_2 = torch.squeeze(show_label_2[0], dim = 0).detach().cpu().numpy()
        # colorized = colorize(show_label_2)
        # og_img_2 = (255*imgs[0]).permute((1, 2, 0)).type(torch.uint8).detach().cpu().numpy()
        # og_img_2 = Image.fromarray(og_img_2)
        # # og_img_2.save('og_img.png')
        # colorized.save('tester_2.png')
        # print('dshjb')
        writer.add_scalar('train_iou', iou.item(), i*len(ds_train) + k)
        writer.add_scalar('train_loss', loss.item(), i*len(ds_train) + k)
        torch.cuda.empty_cache()
        k = k + 1

    scheduler.step()    
    return student

def test_fn(student, ds_test, i):
    resizer_1 = torchvision.transforms.Resize(size = (1024, 1024))
    resizer_2 = torchvision.transforms.Resize(size = (256, 256))
    # teacher.eval()
    student.eval() 
    student.to(device_2)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction = 'mean').to(device_2)
    jaccard = torchmetrics.JaccardIndex(19).to(device_2)
    sigmoid = torch.nn.Sigmoid()
    total_iou = 0
    k = 0
    for imgs in ds_test:
        imgs_2 = imgs.to(device_2)
        imgs_1 = resizer_1(imgs)
        imgs_2 = resizer_1(imgs_2)
        teacher_label, teacher_iou = get_two_classes(teacher(imgs_1)['logits'].to(device_2))
        teacher_label = get_hard_labels(teacher_label)
        student_label, student_iou = get_two_classes(student(imgs_2).to(device_2))
        del imgs_1
        del imgs_2
        student_label = resizer_2(student_label)
        student_iou = resizer_2(student_iou)
        # loss = loss_fn(student_label, sigmoid(teacher_label.detach()))
        loss = loss_fn(student_label, teacher_label.detach().float())
        del teacher_label
        # show_label = teacher_iou
        # show_label_2 = student_iou
        iou = jaccard(teacher_iou, student_iou)
        del teacher_iou
        # print(f'train_loss : {loss.item()} | train_IOU : {iou.item()}')
        print(f'train_IOU : {iou.item()}')
        del student_label
        del student_iou
        total_iou += iou
        print(f'test_loss : {loss.item()} | test_IOU : {iou.item()}')
        writer.add_scalar('test_loss', loss.item(), i*len(ds_test) + k)
        writer.add_scalar('test_iou', iou.item(), i*len(ds_test) + k)
        k = k + 1
    mean_iou = total_iou/len(ds_test)
    # writer.add_scalar('mean_test_iou', mean_iou, i)
    print(f'*************************** | mean_iou : {mean_iou} | ***************************')
    torch.save(student.state_dict(), f'students/{mean_iou}.pt')
    
    return student

def kd(student, ds_train, ds_test, num_epochs):
    t = 0
    for i in range(num_epochs):
        student = train_fn(student, ds_train, t)
        student = test_fn(student, ds_test, t)
        t = t + 1


if __name__ == '__main__':
    stud_name = 'fast_seg' # bisenet, fast_seg
    if stud_name == 'fast_seg':
        student = get_fastseg_model()
    # if stud_name == 'bisenet':
    #     student = get_bisenet()
    # teacher = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024", force_download=False)
    
    ds_train, ds_test = get_only_nsv_360_images_dataloaders(2, True)
    kd(student, ds_train, ds_test, 50)
    writer.close()