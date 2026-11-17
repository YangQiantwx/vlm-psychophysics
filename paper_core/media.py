"""Replay the axis and size interventions and verify lossless decoded RGB."""

from argparse import ArgumentParser
from fractions import Fraction
from hashlib import sha256 as hash_bytes
from pathlib import Path

import av
import cv2
import numpy as np
from PIL import Image

from .scores import read_json, require, sha256

WIDTH, HEIGHT, FPS, FRAMES = 768, 512, 60, 180
T3_CENTERS = ((224, 304), (544, 304))


def pixel_sha(pixels):
    return hash_bytes(np.ascontiguousarray(pixels).tobytes()).hexdigest()


def read_video(path):
    frames = []
    with av.open(str(path)) as media:
        stream, = media.streams.video
        stream.codec_context.thread_count = 1
        require((stream.width, stream.height, stream.average_rate) == (WIDTH, HEIGHT, FPS), path)
        for index, frame in enumerate(media.decode(stream)):
            require(frame.pts * frame.time_base == Fraction(index, FPS), (path,index,'timeline'))
            frames.append(frame.to_ndarray(format='rgb24'))
    require(len(frames) == FRAMES, (path, 'frame count'))
    return frames


def decoded_sha(frames):
    digest = hash_bytes()
    for array in frames:
        digest.update(np.ascontiguousarray(array).data)
    return digest.hexdigest()


def write_video(path, frames, *, t3):
    with av.open(str(path), 'w') as media:
        stream = media.add_stream('libx264rgb', rate=FPS)
        stream.width, stream.height = WIDTH, HEIGHT
        stream.pix_fmt = 'rgb24'
        stream.codec_context.thread_count = 1
        stream.options = {'crf': '0', 'preset': 'ultrafast', 'threads': '1'}
        if t3:
            stream.options.update({'ref':'16','scenecut':'0'})
        for i, array in enumerate(frames):
            frame = av.VideoFrame.from_ndarray(array, format='rgb24')
            frame.pts, frame.time_base = i, Fraction(1, FPS)
            for packet in stream.encode(frame):
                media.mux(packet)
        for packet in stream.encode():
            media.mux(packet)
    require(decoded_sha(read_video(path)) == decoded_sha(frames), (path, 'encoded RGB drift'))


def black_axis(image, inclinations):
    """Apply the same antialiased 3.6 px core and 7.4 px white halo."""
    result = image.copy()
    for (x, y), theta in zip(T3_CENTERS, inclinations, strict=True):
        xp, yp = np.meshgrid(np.arange(x-127,x+128)+.5-x, np.arange(y-127,y+128)+.5-y)
        radians = np.deg2rad(theta)
        along = xp * np.sin(radians) + yp * np.cos(radians)
        across = xp * np.cos(radians) - yp * np.sin(radians)
        longitudinal = np.clip(120.5 - np.abs(along), 0., 1.)
        core = (np.clip(1.8 - np.abs(across), 0., 1.) * longitudinal)[...,None]
        halo = (np.clip(3.7 - np.abs(across), 0., 1.) * longitudinal)[...,None]
        area = result[y-127:y+128, x-127:x+128]
        rim = (1 - halo) * area.astype(np.float32) + halo * 246.
        area[:] = np.rint((1 - core) * rim + core * 8.).astype(np.uint8)
    return result


def angles(row):
    theta = row['guide_theta_deg']
    return (0., theta) if row['truth'] == 'LEFT' else (theta, 0.)


def replay_axis_video(root, row, output):
    path = root/row['base_path']
    require(sha256(path) == row['base_sha256'], (path, 'original SHA-256'))
    original = read_video(path)
    require(decoded_sha(original) == row['base_rgb_sha256'], (path, 'original RGB'))
    on, off = (row[k] for k in ('onset_frame', 'offset_frame'))
    require(off - on == 30, 'T3 event duration drift')
    marked = black_axis(original[on], angles(row))
    rebuilt = [marked if on <= i < off else frame for i, frame in enumerate(original)]
    require(decoded_sha(rebuilt) == row['marked_rgb_sha256'], (path, 'marked RGB mismatch'))
    write_video(output, rebuilt, t3=True)


def linear(value):
    return np.where(value <= .04045, value/12.92, ((value+.055)/1.055)**2.4)


def rgb8(value):
    x = np.clip(value, 0., 1.)
    x = np.where(x <= .0031308, 12.92*x, 1.055*x**(1/2.4)-.055)
    return np.rint(255*x).astype(np.uint8)


def replay_axis_static(root, row, video, patches, output):
    path = root/row['base_path']
    require(sha256(path) == row['input_digests']['base']['sha256'], (path, 'source SHA-256'))
    with Image.open(path) as im:
        source = np.asarray(im.convert('RGB'))
    movie = root/video['base_path']
    require(sha256(movie) == video['base_sha256'], (movie, 'source SHA-256'))
    background = read_video(movie)[video['onset_frame']-1]
    flat = np.full_like(background, 235)
    canvas = linear(flat.astype(np.float32)/255.)
    for name, (x,y) in zip(('LEFT','RIGHT'), T3_CENTERS, strict=True):
        digest = row['geometry'][name]['patch_sha256']
        path = patches/f'{digest}.npy'
        require(sha256(path)==digest,(name,'wrong source alpha patch'))
        patch = np.load(path, allow_pickle=False)
        require(patch.shape==(320,320,4),(name,'wrong source alpha patch shape'))
        roi = canvas[y-160:y+160,x-160:x+160]
        roi[:] = patch[:,:,:3] + (1-patch[:,:,3,None])*roi
    for name, image in {'base':source, 'black_exact':source, 'scene_only':background,
                        'objects_only':rgb8(canvas), 'black_only':flat}.items():
        result = image if name == 'base' else black_axis(image, angles(row))
        require(pixel_sha(result) == row['input_digests'][name]['rgb_sha256'], (name, 'RGB differs'))
        Image.fromarray(result).save(output/f'{name}.png', compress_level=1)


def replay_t5(original, reference, sprite, row, output):
    require(sha256(original) == row['original_video_sha256'], 'Wrong source video')
    require(sha256(reference) == row['original_reference_sha256'], 'Wrong source reference')
    require(sha256(sprite) == row['sprite_sha256'], 'Wrong source sprite')
    frames = read_video(original)
    require(decoded_sha(frames) == row['source_decoded_rgb_sha256'], 'Wrong decoded original')
    # At least one pre-event or post-event frame is always present.
    event_frames = {i for r in row['events'] for i in range(r['event']['onset_frame'], r['event']['onset_frame'] + 30)}
    base = next(frame for i, frame in enumerate(frames) if i not in event_frames)
    require(pixel_sha(base) == row['background_sha256'], 'Wrong scene backdrop')
    width = row['video_sprite_px']
    radius = width // 2
    sprite = cv2.resize(np.load(sprite, allow_pickle=False), (width, width), interpolation=cv2.INTER_LINEAR)
    images = [base] * FRAMES
    for r in row['events']:
        x, y = r['center']
        event = r['event']
        rotated = np.rot90(sprite, -event['angle_deg']//90)
        frame = base.copy()
        area = frame[y-radius:y+radius, x-radius:x+radius]
        back = linear(area.astype(np.float64)/255.)
        area[:] = rgb8(rotated[:,:,:3] + (1-rotated[:,:,3,None])*back)
        on = event['onset_frame']
        images[on:on+30] = [frame] * 30
    require(decoded_sha(images) == row['rgb_sha256'], 'Enlarged video RGB differs')
    with Image.open(reference) as source:
        ref = np.array(source.convert('RGB'))
    center = ((WIDTH-1)/2, (HEIGHT-1)/2)
    factor = row['gray_reference']['factor']
    matrix = cv2.getRotationMatrix2D(center, 0, factor)
    gray = tuple(int(v) for v in ref[0,0])
    new_ref = cv2.warpAffine(ref, matrix, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=gray)
    require(pixel_sha(new_ref) == row['gray_reference']['pixel_sha256'], 'Enlarged reference RGB differs')
    Image.fromarray(new_ref).save(output/'reference.png')
    write_video(output/'video.mp4', images, t3=False)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('t3-video','t3-static','t5-enlarged'))
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--id', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--patches', type=Path)
    parser.add_argument('--original', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--sprite', type=Path)
    args = parser.parse_args()
    if args.mode == 't5-enlarged':
        if any(p is None for p in (args.original, args.reference, args.sprite)):
            parser.error('Provide --original, --reference and --sprite for enlargement')
        row = read_json(args.study/'t5-enlarged-inputs.json')['inputs'][args.id]
        args.out.mkdir(parents=True,exist_ok=True)
        replay_t5(args.original,args.reference,args.sprite,row,args.out)
    else:
        row = read_json(args.study/'t3-axis-inputs.json')['inputs'][args.id]
        if args.mode == 't3-video':
            replay_axis_video(args.base,row,args.out)
        else:
            if args.patches is None:
                parser.error('Provide the alpha patches with --patches')
            static = read_json(args.study/'t3-static-inputs.json')['inputs'][args.id]
            args.out.mkdir(parents=True,exist_ok=True)
            replay_axis_static(args.base,static,row,args.patches,args.out)
    print(f'Verified original and regenerated RGB for {args.id}.')


if __name__ == '__main__':
    main()
