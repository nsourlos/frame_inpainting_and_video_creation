#Segment anything taken from https://github.com/facebookresearch/segment-anything
#Segment based on text taken from https://github.com/luca-medeiros/lang-segment-anything
#Supportive information for GroundingDINO installation at https://github.com/IDEA-Research/GroundingDINO
#Alternative to langsam https://github.com/IDEA-Research/Grounded-Segment-Anything

#Implementation for videos do not work properly (like https://github.com/kadirnar/segment-anything-video)
#or only produce 5frames/second in Hugging Face Spaces (like https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once)

#https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img
#https://lablab.ai/t/stable-diffusion-img2img

#How it works:
# - Loads the first inpainted img
# - Load previous frame and get its mask
# - Load current frame and get its mask
# - Get common pixels of first inpainted img and current frame
# - Replace current frame pixels that match those of first inpainted img with them (strange! Is it like a seed?)
# - Inpaint the rest of the current frame
# - Blend the inpainted img with the original frame                                                                             

#Dependencies
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageFilter
from lang_sam import LangSAM
import copy

#Prompts
text_prompt = "costume" #what to segment - replace below with stable diffusion ('shirt' segments the white between 'costume')
prompt_sd='A fancy summer shirt with flowers in a black background' #what to inpaint

#Paths
sam_checkpoint = "/home/soyrl/sam_vit_b_01ec64.pth" #sam model path
model_type = "vit_b"

frame_path="/home/soyrl/frames/" #Path of all video frames _new_2_upscale
save_path="/home/soyrl/inpainted_imgs/" #Path to save inpainted frames

if not os.path.exists(save_path): #Create folder to save inpainted frames
    os.makedirs(save_path)

#Area to crop to be used as input to stable diffusion
target_width = 1080
target_height = 1080

#Inpainting taken from https://www.youtube.com/watch?v=CERvlvUvVEI&t=34s&ab_channel=AbhishekThakur
pipe=StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting", safety_checker=None)
#torch_dtype=torch.float16 or 32 gives error for my GPU as explained in https://github.com/huggingface/diffusers/issues/2153
#img2img alternative doesn't produce good results
#An alternative model for inpainting is 'runwayml/stable-diffusion-inpainting', but not very good. 

#Activate below to speed up - Same as having device in GPU but better here since both SD and SAM are in CPU
pipe.enable_sequential_cpu_offload()
pipe.enable_attention_slicing("max")

#Initialize langsam model
model = LangSAM(model_type,sam_checkpoint) #Initialize langsam model

#Function to inpaint image - Crop around the center of it first 
def crop_and_resize(image,mask, target_width=1080, target_height=1080):

    image=Image.fromarray(image)
    mask=Image.fromarray(mask)

    target_width =target_width
    target_height = target_height

    # Get the dimensions of the original image
    img_width, img_height = image.size
    
    # Calculate the coordinates to crop around the center
    left = (img_width - target_width) // 2
    upper = (img_height - target_height) // 2
    right = (img_width + target_width) // 2
    lower = (img_height + target_height) // 2
    
    # Crop the image
    cropped_img = image.crop((left, upper, right, lower))
    cropped_mask = mask.crop((left, upper, right, lower))

    image=cropped_img
    mask=cropped_mask

    return image,mask


def inpaint(image,mask,prompt):
        #For consistent image generation each time look at https://huggingface.co/docs/diffusers/using-diffusers/reusing_seeds
        generator = torch.Generator().manual_seed(0) # change the seed to get different results

        # Parameters in pipeline below in https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/inpaint
        sd_image=pipe(prompt=prompt,image=image, mask_image=mask, #num_images_per_prompt=2,#strength=1.0, #guidance_scale=7.5, 
                    generator=generator).images  
        
        return sd_image

#two functions implemented so that we only have inpainting inside the mask as stated in 
# https://github.com/huggingface/diffusers/issues/3514. More details in https://github.com/Markus-Pobitzer/Inpainting-Tutorial/blob/main/InpaintingTutorial.ipynb

# This function blends the inpainted image with the original image using the mask.
# Blur is used to better blend the two images toghether
def blend_image(inpainted, original, mask, blur=3):
    mask = mask.convert("L")
    # Apply blur
    mask = mask.filter(ImageFilter.GaussianBlur(blur))
    # Blend images together
    return Image.composite(inpainted.convert('RGBA'), original.convert('RGBA'), mask).convert('RGBA')


#Loop over each frame
for iter,frame_name in enumerate(sorted(os.listdir(frame_path))):


    print("Processing", frame_name)

    if iter==0: #For the first frame
            
        #Load the first frame, predict masks and convert them to 0-255 uint8
        image = cv2.imread(frame_path+frame_name) #Read image of first frame
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #convert it to RGB

        masks, boxes, phrases, logits = model.predict(Image.fromarray(image), text_prompt) #Predict masks for the first frame
        masks=masks.detach().cpu().numpy() #Convert tensors to numpy and move them to cpu

        masks_copy=copy.deepcopy(masks) #Get a copy of the mask of the first frame
        masks_copy=masks_copy[0] #Get the actual mask of the first frame - Just true/false of 512*512
        masks_copy=masks_copy.astype(np.uint8) #Convert to uint8
        masks_copy=masks_copy*255 #Now values are 0 and 255

        #Crop image and mask around the center and resize them
        frame0_img, frame0_mask=crop_and_resize(image.copy(),masks_copy,target_width=target_width, target_height=target_height) 
        frame0_img=frame0_img.resize((512,512))
        frame0_mask=frame0_mask.resize((512,512))

        sd_image=inpaint(frame0_img, frame0_mask,prompt_sd) #Inpaint the image

        blended_image = blend_image(sd_image[0].copy(), frame0_img,frame0_mask)
    
        blended_image.convert('RGB').save(save_path+frame_name[:-4]+"_inpainted.jpg") #Save the image

    else: #For the rest of the frames

        orig_frame_name=sorted(os.listdir(save_path))[0] #Used just to load the inpainted image below
        inpainted_img=save_path+orig_frame_name
        loaded_inpainted_img=Image.open(inpainted_img) #Load the inpainted image of the previous frame

        image_path=frame_path+frame_name #Path of current frame
        current_image = cv2.imread(image_path) #Image of current frame in full resolution
        current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB) #convert it to RGB  
       
        crop_img,_=crop_and_resize(current_image,current_image,target_width=target_width, target_height=target_height) #Crop around the center of the image
        crop_img=crop_img.resize((512,512)) #Resize it to 512*512 - This is still the original frame

        crop_img=np.asarray(crop_img) #convert cropped original image to array - shape 512*512*3
        crop_img_copy=copy.deepcopy(crop_img) #Get a copy of the cropped image to replace its pixels with the inpainted ones

        #Predict masks for the current frame 
        mask_current_frame, boxes, phrases, logits = model.predict(Image.fromarray(crop_img), text_prompt) 

        mask_current_frame=mask_current_frame.detach().cpu().numpy() #Move the masks to cpu and convert them to numpy

        mask_current_frame=mask_current_frame[0] #Get the actual mask of the original frame
        mask_current_frame=mask_current_frame.astype(np.uint8) #Convert to uint8
        mask_current_frame=mask_current_frame*255 #Now values are 0 and 255

        mask_inpainted = cv2.bitwise_and(np.uint8(loaded_inpainted_img), 
                                    np.uint8(loaded_inpainted_img), 
                                    mask=np.uint8(mask_current_frame)) #Get the inpainted image inside the mask 512*512*3 - used to replace the mask of the next frame

        #Loop over the copy of the cropped image and replace the pixels with the inpainted ones
        for i in range(crop_img_copy.shape[0]): 
            for j in range(crop_img_copy.shape[1]):
                if mask_current_frame[i, j] == 255: # Check if the pixel is white in the black and white mask
                    crop_img_copy[i, j] = mask_inpainted[i, j] #Replace the pixel with the inpainted one - below we will inpaint the rest of the image
                    #This also contains some 'black' pixels from original image since mask was not properly segmented

        #Inpaint the rest of the image
        sd_current_image_crop=inpaint(Image.fromarray(copy.deepcopy(crop_img_copy)),Image.fromarray(mask_current_frame),prompt_sd) 

        #Add inpainted mask in original frame - 'crop_img_copy' will not work
        blended_image = blend_image(sd_current_image_crop[0].copy(), Image.fromarray(crop_img),Image.fromarray(mask_current_frame))
        blended_image.convert('RGB').save(save_path+frame_name[:-4]+"_inpainted.jpg") #Save the image