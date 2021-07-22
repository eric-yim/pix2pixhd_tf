from utils.data_utils import tensor2img
from utils.grid import grid_imgs
from PIL import Image
def save_image(sample_dir,sample_format,epoch,generated):
    gen = tensor2img(generated[0])    
    img = Image.fromarray(gen)
    img_name = sample_dir + sample_format.format(str(epoch).zfill(4))
    img.save(img_name)

def evaluate(model,dataset,sample_dir,sample_format,max_num = 4):
    imgs = []
    for i,(source,target) in enumerate(dataset):
        generated = model.netG.call(source)
        imgs.append(tensor2img(source[0]))
        imgs.append(tensor2img(target[0]))
        imgs.append(tensor2img(generated[0]))

        if (i+1)==max_num:
            break
    img = grid_imgs(imgs,cols=3)
    img = Image.fromarray(img)
    img.save(sample_dir+'eval.png')
    print(f"Samples created: {sample_dir+'eval.png'}")