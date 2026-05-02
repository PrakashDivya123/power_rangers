import voxel_ai_toolkit
op = getattr(voxel_ai_toolkit, 'AnalyzeVideoClip', None)
print('HAS_CLASS', bool(op))
if op:
    inst = op()
    cfg = inst.config
    print('OP_NAME', cfg.name)
    print('OP_LABEL', cfg.label)
