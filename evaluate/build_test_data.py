# This file builds the test set 1 for model evaluation. Test set 1 (TS1) consists of
# 100 clusters of images, 10 for each of the 10 most common objects in Visual Genome,
# namely, man, person, woman, building, sign, table, bus, window, sky, and tree.
# For more details see Cohn-Gordon et al. (2018), p. 4.

import os
import json
import pickle
from PIL import Image as PIL_Image
from utils.image_and_text_utils import vectorize_caption,valid_item,index_to_char,char_to_index,edit_region,get_img_from_id


def get_entries_containing_object(object):
    ''' 
    gets the first 100 datasets that contains a given object in its region description
    returns a id_to_caption disctionary, e.g. {'1_3887': ('tree with sparse leaves', (177.5, 1, 564.5, 388)), ...
    ''' 
    
    # load the region_descriptions file
    json_data=json.loads(open('vg_data/region_descriptions.json','r').read())
    print("READ JSON, len:",len(json_data))

    id_to_caption = dict()
    # iterate over every region of every image 
    for i,image in enumerate(json_data):
        # only store the first 100 entries
        if len(id_to_caption) >= 100:
                break

        for s in image['regions']:
            # break the region description into tokens and 
            # check if the object is in the description
            tokens = s['phrase'].lower().split(' ')
            if object in tokens:

                height = s['height']
                width = s['width']
                sentence = s['phrase'].lower()
                img_id = str(s['image_id'])

                # if the object is found in this region and the entry is valid
                # append this entry to the id_to_caption dictionary
                if valid_item(height,width,sentence,img_id):
                    x_coordinate = s['x']
                    y_coordinate = s['y']                                                
                    region_id = str(s['region_id'])
                    box = edit_region(height,width,x_coordinate,y_coordinate)
                    id_to_caption[img_id+'_'+region_id] = (sentence,box)
                    break

    print(len(id_to_caption))
    return id_to_caption



def make_id_to_caption():
    ''' 
    makes an id_to_caption pickle file that maps image id to its region descriptions
    id_to_caption = {'imageID_regionID': ('vectorizedSentenceDescription', (x1, y1, x2, y2)), ...}
    ''' 
    valids = 0
    invalids = 0
    id_to_caption = {}
    json_data=json.loads(open('vg_data/region_descriptions.json','r').read())
    print("READ JSON, len:",len(json_data))		

    # iterate over every region of every image 
    for i,image in enumerate(json_data):
        for s in image['regions']:

            x_coordinate = s['x']
            y_coordinate = s['y']
            height = s['height']
            width = s['width']
            sentence = s['phrase'].lower()
            img_id = str(s['image_id'])
            region_id = str(s['region_id'])

            # make sure that this data entry is valid
            is_valid = valid_item(height,width,sentence,img_id)

            if is_valid:
                valids+=1
                # calculate the bounding box of this region, box = (x1, y1, x2, y2)
                box = edit_region(height,width,x_coordinate,y_coordinate)
                id_to_caption[img_id+'_'+region_id] = (vectorize_caption(sentence),box)
                # e.g.: id_to_caption['10_1382'] = tuple( vectorize('the boy with ice cream'), (139,82,421,87) )
            else: invalids+=1

        if i%1000==0 and i>0:
            print("PROGRESS:",i)

        # only process the first 6000 images
        if i >6000:
            break

    print(len(id_to_caption))
    print("num valid/ num invalid",valids,invalids) # valids = 22344, invalids = 352129
    print(id_to_caption)
    pickle.dump(id_to_caption,open('vg_data/id_to_caption','wb'))



def crop_image_and_save(id, box, desti_path):
    ''' 
    crop out a region from the image and save it in the provided path
    function parameter: 
        id, box: key, value of an "id_to_caption" pickle file entry
        path: the path to save the cropped image
    '''
    img_id,region_id = id.split('_')
    # get image from path
    path = 'vg_data/VG_100K_2/'+str(img_id)+".jpg"
    img = PIL_Image.open(path)
    # crop region from image
    cropped_img = img.crop(box)
    # resize into square
    resized_img = cropped_img.resize([224,224],PIL_Image.ANTIALIAS)
    # save cropped image into given path
    resized_img.save(desti_path, "JPEG")
    # close img
    img.close()


def build_img_path(dir_path):
    '''
    returns a dictionary of image paths, that in which each category (e.g. tree)
    corresponds to 10 clusters x 10 images, e.g.:
    {
        'window': 
            {
            0: ['vg_data/ts1_img/window/538_4955724', ...],
            1: [..., ...],
            ...
            9: [...]
            }
        'tree': {...}
        ...
    }
    '''
    # initialize empty dictionary
    img = dict()
    # iterate over the image directory
    for dir in os.listdir(dir_path):
        # keep records of clusters
        cluster_idx = 0     
        img[dir] = dict()
        img[dir][cluster_idx] = []
        count = 0
        # append every image path to the corresponding cluster dictionary
        for filename in os.listdir(dir_path + dir):
            img[dir][cluster_idx].append(dir_path + dir + "/" + filename)
            count += 1
            # make a new cluster if there are 10 images in the current cluster
            if count == 10:
                cluster_idx += 1
                img[dir][cluster_idx] = []
                count = 0
            
    return img

    

if __name__ == "__main__":
    # build test dataset 1
    #for object in ["man", "person", "woman", "building", "sign", "table", "bus", "window", "sky", "tree"]:
    #    pickle.dump(get_entries_containing_object(object), open('vg_data/ts1/'+object,'wb'))

    # inspect the pickle file
    #id_to_caption = pickle.load(open("vg_data/ts1/woman",'rb'))
    #print(list(id_to_caption.items())[:10])
    
    # crop out and save the corresponding image regions for testing use
    for filename in os.listdir("vg_data/ts1"):
        id_to_image = pickle.load(open("vg_data/ts1/" + filename, "rb"))
        for id, caption in id_to_image.items():
            crop_image_and_save(id, caption[1], "vg_data/ts1_img/"+filename+"/"+id)
