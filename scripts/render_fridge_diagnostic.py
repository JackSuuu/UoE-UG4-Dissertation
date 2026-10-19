"""Label a single-arm R1 engineering/actor video without inventing a comparison."""
import argparse
import json
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main(run,out):
    status=json.loads((run/'status.json').read_text())
    rows=[json.loads(l) for l in (run/'steps.jsonl').read_text().splitlines()]
    mode=status['args']['mode']
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    title='SCRIPTED PHYSICAL FEASIBILITY (not VLA)' if mode=='servo' else 'FROZEN VLA BASIC-SKILL DIAGNOSTIC'
    out.parent.mkdir(parents=True,exist_ok=True)
    with imageio.get_reader(run/'raw.mp4') as reader, imageio.get_writer(out,fps=20) as writer:
        for i,frame in enumerate(reader):
            row=rows[min(i,len(rows)-1)]
            canvas=Image.new('RGB',(640,640),(20,24,30))
            canvas.paste(Image.fromarray(frame).resize((512,512)),(64,64))
            draw=ImageDraw.Draw(canvas)
            draw.text((16,10),title,font=font,fill='white')
            draw.text((16,36),'Simulation replay 1x; not a real-time or full-task result',font=small,fill='#ffce73')
            draw.text((16,582),f"Phase: {row['phase']} | grasp: {row['grasped']}",font=font,fill='white')
            draw.text((16,610),f"Sim time: {i/20:.2f}s | door: {np.rad2deg(row['door_angle_rad']):.1f} deg"+
                      (' | END FRAME' if i==len(rows) else ''),font=small,fill='white')
            writer.append_data(np.asarray(canvas))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('run',type=Path)
    p.add_argument('out',type=Path)
    a=p.parse_args()
    main(a.run,a.out)
