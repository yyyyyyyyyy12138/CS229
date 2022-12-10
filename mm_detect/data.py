import momaapi
import pprint
import json

dir_moma = "../data/moma_dataset"
# moma = momaapi.MOMA(dir_moma)
moma = momaapi.MOMA(dir_moma)

# get act ids that correspond to just basketball 
# , cnames_act="basketball game"
act_ids_val = moma.get_ids_act()
print(len(act_ids_val))

# anns_acts = moma.get_anns_act(act_ids_val)

# get the ids_hoi of the basketball acts 
# ids_hoi = moma.get_ids_hoi(split="val", ids_act=act_ids_val )
# print(len(ids_hoi))
ids_hoi = moma.get_ids_hoi(ids_act=act_ids_val )

print(len(ids_hoi))

# anns_act = moma.get_anns_act(ids_act)
# print(anns_acts)

def create_dataset(moma, ids_hoi, kind, cname_to_cid):
    records = []

    # coco dataset format 
    coco = {
        'images': [],
        'annotations': [],
        'categories': []
    }

    cnames_id = {}
    id_cnames = {}
    id = 0

    # annotations also need id 
    bbox_id = 0

    for id_hoi in ids_hoi:
        ann_hoi = moma.get_anns_hoi([id_hoi])[0]
        image_path = moma.get_paths(ids_hoi=[id_hoi])[0]
        id_act = moma.get_ids_act(ids_hoi=[id_hoi])[0]
        metadatum = moma.get_metadata(ids_act=[id_act])[0]

        if kind is None:
            entities = ann_hoi.actors + ann_hoi.objects
        elif kind == "actor":
            entities = ann_hoi.actors
        else:  # kind == 'object'
            entities = ann_hoi.objects

        # annotations = {}
        # looping through bounding boxes 
        for entity in entities:

            annotation = {
                'is_crowd': 0,
                'image_id':  ann_hoi.id, 
                'id':bbox_id
                }

            # make sure we have id for current cname 
            if entity.cname not in cnames_id:
                cnames_id[entity.cname] = id
                id_cnames[id] = entity.cname
                id += 1

            annotation['category_id'] = cnames_id[entity.cname]

            annotation['bbox'] = [
                        entity.bbox.x,
                        entity.bbox.y,
                        entity.bbox.width,
                        entity.bbox.height]

            annotation['area'] = entity.bbox.width * entity.bbox.height

            bbox_id += 1

            coco['annotations'].append(annotation)

        image = {
            'file_name' : image_path,
            'height' : metadatum.height,
            'width' : metadatum.width,
            'id' :  ann_hoi.id
        }

        coco['images'].append(image)
        # records.append(record)

        # print(dir(metadatum)) 
        # print(dir(ann_hoi))
        # break

    for category in cnames_id:
        coco['categories'].append({'id': cnames_id[category], 'name': category})

    with open("cnames_id_entire_dataset.json", "w") as outfile:
        json.dump(cnames_id, outfile)

    # print("Cnames")
    # print(cnames)
    # return records
    return coco

dataset = create_dataset(moma=moma,ids_hoi=ids_hoi,kind=None,cname_to_cid=None)
# # pprint.pprint(dataset)

# with open("train.json", "w") as outfile:
#     json.dump(dataset, outfile)