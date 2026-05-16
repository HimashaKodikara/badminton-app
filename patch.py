import re

files = ['TrackNetV3/predict.py', 'TrackNetV3/test.py']

for path in files:
    with open(path, 'r') as f:
        code = f.read()

    # Replace .cuda() with .cpu()
    code = code.replace('.cuda()', '.cpu()')
    code = code.replace('.cuda(device)', '.cpu()')

    # Replace map_location cuda with cpu
    code = code.replace("map_location='cuda'", "map_location='cpu'")
    code = code.replace('map_location="cuda"', 'map_location="cpu"')

    # Replace torch.load calls missing map_location
    code = re.sub(
        r'torch\.load\(([^)]+)\)(?!\s*,\s*map_location)',
        lambda m: 'torch.load(' + m.group(1) + ', map_location="cpu", weights_only=False)',
        code
    )

    # Fix num_workers for CPU
    code = code.replace(
        'num_workers = args.batch_size if args.batch_size <= 16 else 16',
        'num_workers = 0'
    )

    with open(path, 'w') as f:
        f.write(code)
    print('Patched:', path)

print('All done! Verifying...')

for path in files:
    with open(path) as f:
        lines = f.readlines()
    hits = [(i+1, l.strip()) for i, l in enumerate(lines) if '.cuda' in l]
    print(path, ':', len(hits), 'cuda refs remaining')
    for ln, txt in hits:
        print('  line', ln, ':', txt)