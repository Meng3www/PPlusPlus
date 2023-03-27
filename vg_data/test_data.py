import json
import pickle
from PIL import Image as PIL_Image
from utils.image_and_text_utils import vectorize_caption,valid_item,index_to_char,char_to_index,edit_region,get_img_from_id

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



def get_region_from_image(id, caption_vector):
    ''' 
    returns a region cropped from the image
    function parameter: id, caption_vector: key, value of a "id_to_caption" pickle file entry
    '''
    img_id,region_id = id.split('_')
    # get image from path
    path = 'vg_data/VG_100K_2/'+str(img_id)+".jpg"
    img = PIL_Image.open(path)
    #crop region from image
    box = caption_vector[1]
    cropped_img = img.crop(box)
    #resize into square
    resized_img = cropped_img.resize([224,224],PIL_Image.ANTIALIAS)

    return resized_img
    


if __name__ == "__main__":
    # print("MAKING id_to_caption")
    # make_id_to_caption()

    id_to_caption = pickle.load(open("vg_data/id_to_caption",'rb'))
    print("len id_to_caption",len(id_to_caption))

    for id, caption_vector in id_to_caption.items():
        # do something
        pass