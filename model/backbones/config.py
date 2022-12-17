create_model = {
    "efficientnetv2_rw_t": 'timm.create_model("efficientnetv2_rw_t", in_chans=args.c_in)',
    "gc_efficientnetv2_rw_t": 'timm.create_model("gc_efficientnetv2_rw_t", in_chans=args.c_in)',
    "efficientnetv2_s": 'timm.create_model("efficientnetv2_s", in_chans=args.c_in)',
    'efficientnetv2_rw_m': 'timm.create_model("efficientnetv2_rw_m", in_chans=args.c_in)',
}

checkpoint = {
    "efficientnetv2_rw_t": 'ImageModels/efficientnetv2_rw_t.pth',
    "gc_efficientnetv2_rw_t": 'ImageModels/gc_efficientnetv2_rw_t.pth',
    "efficientnetv2_rw_s": 'ImageModels/efficientnetv2_rw_s.pth',
    "efficientnetv2_rw_m": 'ImageModels/efficientnetv2_rw_m.pth'
}

output_channel = {
    "efficientnetv2_rw_t": 1024,
    "gc_efficientnetv2_rw_t": 1024,
    "efficientnetv2_rw_s": 1792,
    "efficientnetv2_rw_m": 2152,
}