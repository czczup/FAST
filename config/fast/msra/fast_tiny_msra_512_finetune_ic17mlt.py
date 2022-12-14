model = dict(
    type='FAST',
    backbone=dict(
        type='fast_backbone',
        config='config/fast/nas-configs/fast_tiny.config'
    ),
    neck=dict(
        type='fast_neck',
        config='config/fast/nas-configs/fast_tiny.config'
    ),
    detection_head=dict(
        type='fast_head',
        config='config/fast/nas-configs/fast_tiny.config',
        pooling_size=7,
        dropout_ratio=0.1,
        loss_text=dict(
            type='DiceLoss',
            loss_weight=0.5
        ),
        loss_kernel=dict(
            type='DiceLoss',
            loss_weight=1.0
        ),
        loss_emb=dict(
            type='EmbLoss_v2',
            feature_dim=4,
            loss_weight=0.25
        )
    )
)
repeat_times = 10
data = dict(
    batch_size=16,
    train=dict(
        type='FAST_MSRA',
        split='train',
        is_transform=True,
        img_size=512,
        short_size=512,
        pooling_size=7,
        read_type='cv2',
        repeat_times=repeat_times
    ),
    test=dict(
        type='FAST_MSRA',
        split='test',
        short_size=512,
        read_type='pil'
    )
)
train_cfg = dict(
    lr=1e-3,
    schedule='polylr',
    epoch=600 // repeat_times,
    optimizer='Adam',
    pretrain='pretrained/fast_tiny_ic17mlt_640.pth',
    # https://github.com/czczup/FAST/releases/download/release/fast_tiny_ic17mlt_640.pth
    save_interval=10 // repeat_times,
)
test_cfg = dict(
    min_score=0.88,
    min_area=200,
    bbox_type='rect',
    result_path='outputs/submit_msra/'
)
