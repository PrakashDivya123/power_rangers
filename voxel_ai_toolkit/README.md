# Video Intelligence Hub

A FiftyOne plugin providing an interactive hub for dataset curation and anomaly review using Twelve Labs (Pegasus and Marengo).

## Installation

You can install this plugin into your local FiftyOne app by running:
```bash
fiftyone plugins download d:\VoxelHackathon\voxel_ai_toolkit
```

Or by simply symlinking it into your plugins directory:
```bash
ln -s d:\VoxelHackathon\voxel_ai_toolkit ~/.fiftyone/plugins/voxel_ai_toolkit
```

## Features

- **Analyze Video Clip**: Right-click or use the operators menu on selected video clips to quickly generate Pegasus summaries or find similar anomalies using Marengo indexing.

## Requirements

```bash
pip install fiftyone twelvelabs
```
