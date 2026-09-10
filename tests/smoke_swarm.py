"""Generate five controlled comparisons through a running SwarmUI instance."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urljoin

from PIL import Image, ImageChops, ImageStat
import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:7801')
    parser.add_argument('--model', help='Model name as reported by ListT2IParams')
    parser.add_argument('--preset', help='An existing preset to apply unchanged')
    parser.add_argument('--backend', default='0', help='Exact backend ID to test')
    parser.add_argument('--zero-tolerance', type=float, default=1, help='Maximum mean pixel difference on 0–255 scale; use 0 for bit-exact checks')
    parser.add_argument('--reference', type=Path, default=Path(__file__).parent / 'results/reference-and-pose.png')
    parser.add_argument('--pose', type=Path, default=Path(__file__).parent / 'results/reference-and-pose.png')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'output')
    args = parser.parse_args()
    if not args.model and not args.preset:
        parser.error('Provide --model or --preset.')
    args.output.mkdir(parents=True, exist_ok=True)
    url = args.url.rstrip('/') + '/'
    session = requests.post(url + 'API/GetNewSession', json={}, timeout=30).json()['session_id']

    def post(route, params, timeout=120):
        response = requests.post(url + 'API/' + route, json=dict(session_id=session, **params), timeout=timeout)
        response.raise_for_status()
        result = response.json()
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result

    def data(path):
        return 'data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode()

    params = dict(prompt='1girl, solo, adult woman, full body, front view, standing, wearing a blue blazer over a buttoned white shirt, black suit trousers, brown hair, gray background',
                  negativeprompt='low quality, blurry, deformed, text, watermark', width=1024, height=1024,
                  steps=30, cfgscale=4.5, sampler='euler', scheduler='normal', seed=90210,
                  images=1, exactbackendid=args.backend, donotsave=False, nopreviews=True)
    if args.model:
        params['model'] = args.model
    if args.preset:
        params['presets'] = [args.preset]
    reference, pose = data(args.reference), data(args.pose)
    records = []
    for case, controls in [
        ('baseline', {}),
        ('reference', dict(promptimages=[reference], referencestrength=1)),
        ('pose', dict(poseimage=pose, posestrength=1)),
        ('combined', dict(promptimages=[reference], poseimage=pose, referencestrength=1, posestrength=1)),
        ('zero', dict(promptimages=[reference], poseimage=pose, referencestrength=0, posestrength=0)),
    ]:
        request = params | controls
        graph = post('ComfyGetGeneratedWorkflow', request)['workflow']
        (args.output / f'{case}.workflow.json').write_text(graph, encoding='utf-8')
        if case in ['baseline', 'zero']:
            assert not any(n['class_type'].startswith('SwarmAnima') for n in json.loads(graph).values()), case
        print('Generating', case, flush=True)
        started = time.time()
        result = post('GenerateText2Image', request, timeout=1800)
        src = result['images'][0]
        src = src if isinstance(src, str) else src['src']
        response = requests.get(urljoin(url, src), timeout=120)
        response.raise_for_status()
        path = args.output / f'{case}.png'
        path.write_bytes(response.content)
        digest = hashlib.sha256(Image.open(path).convert('RGB').tobytes()).hexdigest()
        records.append(dict(case=case, pixel_sha256=digest, seconds=round(time.time() - started, 2)))
        (args.output / 'checks.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    baseline = Image.open(args.output / 'baseline.png').convert('RGB')
    zero = Image.open(args.output / 'zero.png').convert('RGB')
    error = sum(ImageStat.Stat(ImageChops.difference(baseline, zero)).mean) / 3
    comparison = dict(bit_identical=records[0]['pixel_sha256'] == records[-1]['pixel_sha256'], mean_pixel_difference=error)
    (args.output / 'zero-comparison.json').write_text(json.dumps(comparison, indent=2), encoding='utf-8')
    assert error <= args.zero_tolerance, f'Disabling controls changed baseline by {error:.3f}/255.'
    assert all(r['pixel_sha256'] != records[0]['pixel_sha256'] for r in records[1:-1]), 'A conditioning path had no effect.'
    print(f'PASS: both paths affect output; zero strength differs by {error:.3f}/255. Inspect images for guidance quality.')


if __name__ == '__main__':
    main()
