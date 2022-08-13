import torch
from fast_seg import get_colorized_seg_map, get_fastseg_model
from get_segformer import get_segformer, get_segformer_feature_extractor
from fastseg.image import colorize, blend
from get_dataloader import get_only_nsv_360_images_dataloaders, get_rdd_dataloaders
import torchvision
import torchmetrics
from PIL import Image 

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
segformer = get_segformer().to(device)
segformerextractor = get_segformer_feature_extractor()
segformer.eval()

def teacher(inputs):
    im_list = [inputs[x] for x in range(inputs.shape[0])]
    inputs = segformerextractor(images = im_list, return_tensors="pt")['pixel_values'].to(device) 
    return segformer(inputs)

def get_two_classes(vector):
    total = torch.cat((torch.unsqueeze(vector[:, 0, :, :], dim = 1), torch.sum(vector, dim = 1, keepdim = True) - torch.unsqueeze(vector[:, 0, :, :], dim = 1)), dim = 1)
    arg = torch.argmax(vector, dim = 1, keepdim = True)
    arg[arg != 0] = 1
    arg[arg == 0] = 0
    return total, arg


def visualize(student, ds):
    resizer_1 = torchvision.transforms.Resize(size = (1024, 1024))
    resizer_2 = torchvision.transforms.Resize(size = (128, 128))
    student.to(device)
    student.train() 
    jaccard = torchmetrics.JaccardIndex(19).to(device)
    for imgs in ds:
        torch.cuda.empty_cache()
        imgs = resizer_1(imgs)
        waste, teacher_iou = get_two_classes(teacher(imgs)['logits'].to(device))
        teacher_label = torch.permute(torch.squeeze(torch.nn.functional.one_hot(teacher_iou, num_classes=2)), (0, 3, 1, 2))
        imgs = imgs.to(device)
        student_label, student_iou = get_two_classes(student(imgs).to(device))
        student_label = resizer_2(student_label)
        student_iou = resizer_2(student_iou)
        del teacher_label
        show_label = teacher_iou
        show_label_2 = student_iou
        iou = jaccard(teacher_iou, student_iou)
        del student_iou
        del teacher_iou
        print(f'IOU : {iou.item()}')
        del student_label
        show_label = torch.squeeze(show_label[0], dim = 0).detach().cpu().numpy()
        colorized_1 = colorize(show_label)
        og_img = (255*imgs[0]).permute((1, 2, 0)).type(torch.uint8).detach().cpu().numpy()
        og_img = Image.fromarray(og_img)
        og_img.save('og_img.png')
        colorized_1.save('teacher_preds.png')
        show_label_2 = torch.squeeze(show_label_2[0], dim = 0).detach().cpu().numpy()
        colorized_2 = colorize(show_label_2)
        og_img_2 = (255*imgs[0]).permute((1, 2, 0)).type(torch.uint8).detach().cpu().numpy()
        og_img_2 = Image.fromarray(og_img_2)
        colorized_2.save('student_preds.png')
        exit()
    
if __name__ == '__main__':
    ds, ds_2 = get_rdd_dataloaders(3, True)
    # ds, ds_2 = get_only_nsv_360_images_dataloaders(3, True)
    student  = torch.load("students/0.10203295201063156.pt")
    visualize(student, ds)