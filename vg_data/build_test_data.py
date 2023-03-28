# This file builds the test set 1 for model evaluation. Test set 1 (TS1) consists of
# 100 clusters of images, 10 for each of the 10 most common objects in Visual Genome,
# namely, man, person, woman, building, sign, table, bus, window, sky, and tree.
# For more details see Cohn-Gordon et al. (2018), p. 4.

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
            if len(id_to_caption) >= 100:
                break

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



def get_region_from_image(id, box):
    ''' 
    returns a region cropped from the image
    function parameter: id, caption_vector: key, value of a "id_to_caption" pickle file entry
    '''
    img_id,region_id = id.split('_')
    # get image from path
    path = 'vg_data/VG_100K_2/'+str(img_id)+".jpg"
    img = PIL_Image.open(path)
    # crop region from image
    # box = caption_vector[1]
    cropped_img = img.crop(box)
    # resize into square
    resized_img = cropped_img.resize([224,224],PIL_Image.ANTIALIAS)
    resized_img.show()

    return resized_img
    


if __name__ == "__main__":
    # print("MAKING id_to_caption")
    # make_id_to_caption()

    # id_to_caption = pickle.load(open("vg_data/id_to_caption",'rb'))
    # print("len id_to_caption",len(id_to_caption))

    # for id, caption_vector in id_to_caption.items():
        # do something
    #    pass


    # pickle.dump(get_entries_containing_object('man'), open('vg_data/ts1/man','wb'))
    # pickle.dump(get_entries_containing_object('person'), open('vg_data/ts1/person','wb'))
    # pickle.dump(get_entries_containing_object('woman'), open('vg_data/ts1/woman','wb'))
    # pickle.dump(get_entries_containing_object('building'), open('vg_data/ts1/building','wb'))
    # pickle.dump(get_entries_containing_object('sign'), open('vg_data/ts1/sign','wb'))
    # pickle.dump(get_entries_containing_object('table'), open('vg_data/ts1/table','wb'))
    # pickle.dump(get_entries_containing_object('bus'), open('vg_data/ts1/bus','wb'))
    # pickle.dump(get_entries_containing_object('window'), open('vg_data/ts1/window','wb'))
    # pickle.dump(get_entries_containing_object('sky'), open('vg_data/ts1/sky','wb'))
    # pickle.dump(get_entries_containing_object('tree'), open('vg_data/ts1/tree','wb'))

    # inspect the pickle file
    id_to_caption = pickle.load(open("vg_data/ts1/tree",'rb'))
    print(list(id_to_caption.items())[:10])

    # inspect the image region
    get_region_from_image('149_4936249', (181, 1, 798, 618))