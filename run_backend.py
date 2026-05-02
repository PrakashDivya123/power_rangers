import os
from pathlib import Path

os.environ['TWELVELABS_API_KEY'] = 'tlk_3W40MP0080RKRJ2C69TG91VCFYGY'
ffmpeg_bin = r'D:\VoxelHackathon\ffmpeg_temp\ffmpeg-8.1-essentials_build\bin'
os.environ['PATH'] = ffmpeg_bin + os.pathsep + os.environ.get('PATH','')

video = r'D:\UX\FigmamakeClaudecode_20260325-2255-17.4396814.mp4'
output = r'.\backend_output'
baseline_dir = r'D:\VoxelHackathon\anomaly_baseline-20260403T193608Z-3-001\anomaly_baseline'

from voxel_ai_toolkit import backend

res = backend.process_video(
    video_path=video,
    output_dir=output,
    api_key=os.environ['TWELVELABS_API_KEY'],
    segment_seconds=10,
    baseline_dir=baseline_dir,
    top_k=3,
)

print('PROCESS_RESULT', res)
