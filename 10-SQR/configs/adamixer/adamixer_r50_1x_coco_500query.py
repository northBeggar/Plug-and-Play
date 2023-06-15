_base_ = './adamixer_r50_1x_coco.py'
num_query = 500
model = dict(
    rpn_head=dict(num_query=num_query),
    test_cfg=dict(
        _delete_=True, rpn=None, rcnn=dict(max_per_img=num_query)))
