import json

# The new config inherits a base config to highlight the necessary modification
_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# mm_det/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py

# with open('../cnames_id_entire_dataset.json') as json_file:
#     cname_to_cid = json.load(json_file)

# num_classes = len(cname_to_cid)

runner = dict(type="EpochBasedRunner", max_epochs=1)

num_classes = 251

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
   roi_head=dict(
       bbox_head=dict(
            type="Shared2FCBBoxHead",
            num_classes=num_classes,
        ),
    )
)

# Modify dataset related settings
#dataset_type = 'COCODataset'
classes = ('babysitter', 'unclassified child', 'bicycle', 'unclassified adult', 'car', 'clothes', 'garbage', 'scooter', 'backpack', 'fan', 'cup/glass', 'unsure', 'table', 'chair', 'police officer', 'smoke', 'towel', 'health worker', 'clip', 'blanket', 'tie', 'security tray', 'handbag', 'paper', 'box', 'metal detector', 'luggage', 'walk-through metal detector', 'book', 'bottle', 'nail polish bottle', 'earphones', 'shoe', 'portable charger', 'laptop', 'cap', 'cell phone', 'crowd', 'customer', 'wallet', 'waiter', 'cashier', 'cup', 'cabinet', 'pen', 'crowbar', 'patient', 'bed', 'wheelchair', 'pillow', 'walking stick', 'board', 'basketball hoop', 'basketball', 'alcohol detector', 'syringe', 'medical cotton', 'soccer player', 'soccer ball', 'soccer gate', 'match official', 'disposable gloves', 'medical tape', 'thermometer', 'bowl', 'bucket', 'spoon', 'ice cube', 'notebook', 'flag', 'water', 'toy', 'carpet', 'stroller', 'sofa', 'coin', 'ball', 'card', 'presenter', 'diaper', 'toy car', "child's chair", 'food', 'basket', 'door', 'alcohol pad', 'computer', 'pad', 'microphone', 'table tennis player', 'bath bucket', 'baby bottle', 'uncertain baby bottle', 'stool/bench', 'gym equipment', 'fork', 'plate', 'receptionist', 'sock', 'pill case', 'packaging bag', 'unclassified athlete', 'pit crew member', 'motorcycle', 'driver', 'camera', 'fence', 'video camera', 'wrench', 'basketball player', 'table tennis ball', 'basketball stand', 'barber', 'razor', 'mirror', 'brush', 'comb/hair brush', 'scissor', 'shower ', 'lotion', 'kettle', 'whistle', 'hair dryer', 'massage chair', 'salon worker', 'hat', 'mouse', 'frisbee player', 'window', 'plastic wrap', 'shampoo', 'glasses', 'lid', 'dining table', 'baby basket', 'floor', 'cloth', 'pot', 'firefighter', 'water pipe', 'animal', 'fire extinguisher', 'firetruck', 'nail clippers', 'nail file', 'car seat', 'medicine bottle', 'gloves', 'reception desk', 'telephone', 'electric screwdriver', 'tire', 'ring box', 'bottle cap', 'table tennis table', 'table tennis racket', 'steering wheel', 'scoretable', 'ring', 'screen', 'house', 'scoreboard', 'security belt', 'sack', 'laying chair', 'key', 'flashlight', 'light', 'toothbrush', 'scanner', 'belt', 'cotton swab', 'bench', 'massage table', 'breathalyzer', 'frisbee', 'umbrella', 'award winner', 'trophy', 'podium', 'microphone stand', 'platter', 'awards podium', "child's bicycle", 'handcuffs', 'file', 'nail polish brush', 'cash', 'card recorder', 'tablet', 'digital device', 'water barrel', 'ladder', 'money', 'walkie-talkie', 'hiking stick', 'wine glass', 'knife', 'wine bottle', 'menu', 'barrel', 'utensil', 'water cup/bottle', 'alcohol', 'shot glass', 'order machine', 'chopsticks', 'utensils', 'decanter', 'sponge', 'drawer', 'receiver ', 'score book', 'cotton balls', 'tweezer', 'mask', 'needle', 'electric saw', 'big car', 'horn', 'guitar', 'shield', 'sword', 'piano', 'stick', 'bread', 'teacher', 'document', 'poker card', 'basin', 'cash register', 'goods', 'straw', 'arena', 'flower', 'stereo', 'quilt', 'cushion', 'toe separator', 'phototherapy machine', 'student', 'coffee table', 'piano book', 'sheet music', 'stool', 'timer', 'yellow card', 'platform', 'racket', 'racer', 'gas pump')
#classes = tuple(cname_to_cid.keys())

data = dict(
   train=dict(
       img_prefix='',
       classes=classes,
       ann_file='../mm_det/train.json'),
   val=dict(
       img_prefix='',
       classes=classes,
       ann_file='../mm_det/val.json'),
   test=dict(
       img_prefix='',
       classes=classes,
       ann_file='../mm_det/test.json'))

# We can use the pre-trained Faster RCNN model to obtain higher performance
load_from = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

checkpoint_config = dict(interval=1)

evaluation = dict(interval=5, metric='bbox')
