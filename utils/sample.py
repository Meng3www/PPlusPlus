import torch
from torch.autograd import Variable
from PIL import Image
import re

def to_var(x, volatile=False):  # tensor (1, 3, 224, 224)
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def load_image_from_path(path, transform=None):
    from PIL import Image as PIL_Image
    with Image.open(path) as image:
        image = image.resize([224, 224], Image.LANCZOS)
        # image = image.crop([0,0,224,224])
        if transform is not None:
            image = transform(image).unsqueeze(0)
        return image

def load_image(url, transform=None):

    # import urllib.request
    # from PIL import Image as PIL_Image
    import shutil
    import requests

    hashed_url = re.sub('/','',url)
    hashed_url = re.sub(':','',hashed_url)

    response = requests.get(url, stream=True)
    # print(response.status_code == 200)
    with open('data/google_images/'+hashed_url+'.jpg', 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
    # del response
    print(url,response)
    # print(os.listdir())

    image = Image.open('data/google_images/'+hashed_url+'.jpg')
    # print("image loaded (sample.py)")
    image = image.resize([224, 224], Image.LANCZOS)
    # width = image.size[0]
    # height = image.size[1]
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    # image = transforms.ToTensor()(image).unsqueeze(0)
    # print(image.shape)
    return image  # Tensor: (1, 3, 224, 224)
    
