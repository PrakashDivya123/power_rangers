"""
Video Sentinel — FiftyOne Plugin
=================================
Temporal anomaly detection, crime classification, and scene description
for surveillance video using Twelve Labs Marengo + Pegasus.

Operators:
    1. detect_anomaly_windows  → WHEN does the crime happen? (10s temporal detections)
    2. classify_crime          → WHAT type of crime is it?
    3. describe_scene          → WHO is involved and what's happening?
"""

import numpy as np
import fiftyone as fo
import fiftyone.operators as foo
from fiftyone import ViewField as F


# ═══════════════════════════════════════════════════════════════════
# Placeholder functions — replace these with real Twelve Labs calls
# ═══════════════════════════════════════════════════════════════════

def get_video_duration(sample):
    """Get video duration in seconds from FiftyOne sample metadata."""
    if sample.metadata and sample.metadata.total_frame_count and sample.metadata.frame_rate:
        return sample.metadata.total_frame_count / sample.metadata.frame_rate
    return 60.0  # fallback


def compute_segment_embeddings(video_path, segment_duration=10.0):
    """
    PLACEHOLDER — Replace with Twelve Labs Marengo embed v2.

    Should:
        1. Call embed.v_2.tasks.create with start_sec/end_sec for each window
           OR use VideoSegmentation_Fixed(duration_sec=segment_duration)
        2. Return list of (start_sec, end_sec, embedding_vector) tuples

    For now: returns random 1024-d vectors for each 10s window.
    """
    # TODO: Replace with real Marengo call
    # from twelvelabs import TwelveLabs, VideoInputRequest, MediaSource
    # client = TwelveLabs(api_key=os.getenv("TWELVE_LABS_API_KEY"))
    # task = client.embed.v_2.tasks.create(
    #     input_type="video",
    #     model_name="marengo3.0",
    #     video=VideoInputRequest(
    #         media_source=MediaSource(base_64_string=video_b64),
    #         segmentation=VideoSegmentation_Fixed(
    #             fixed=VideoSegmentationFixedFixed(duration_sec=segment_duration)
    #         ),
    #     ),
    # )
    # ... poll and extract task.data ...

    duration = 60.0  # placeholder
    segments = []
    t = 0.0
    while t < duration:
        end = min(t + segment_duration, duration)
        embedding = np.random.randn(1024).tolist()
        segments.append((t, end, embedding))
        t = end
    return segments


def compute_normal_baseline(dataset, normal_label="Normal", embedding_field="embedding"):
    """
    PLACEHOLDER — Compute mean embedding from samples labeled as normal.

    Should:
        1. Gather all embeddings from samples with ground_truth == normal_label
        2. Return the mean vector as the "normal" centroid

    For now: returns a random 1024-d vector.
    """
    # TODO: Replace with real aggregation
    # normal_view = dataset.match(F("ground_truth.label") == normal_label)
    # embeddings = np.array([s[embedding_field] for s in normal_view])
    # return embeddings.mean(axis=0)

    return np.random.randn(1024)


def cosine_distance(a, b):
    """Cosine distance between two vectors. 0 = identical, 2 = opposite."""
    a, b = np.array(a), np.array(b)
    sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    return 1.0 - sim


def classify_with_text_embeddings(video_embedding, categories):
    """
    PLACEHOLDER — Replace with Twelve Labs Marengo text embedding.

    Should:
        1. For each category, call client.embed.create(model_name="marengo3.0", text=category)
        2. Compute cosine similarity between video_embedding and each text embedding
        3. Return (best_label, confidence_score)

    For now: returns a random category with a random score.
    """
    # TODO: Replace with real Marengo text embed
    # text_emb = client.embed.v_2.create(
    #     input_type="text",
    #     model_name="marengo3.0",
    #     text=TextInputRequest(content=category),
    # )
    # sim = cosine_similarity(video_embedding, text_emb.data[0].embedding)

    idx = np.random.randint(len(categories))
    confidence = round(np.random.uniform(0.5, 0.99), 3)
    return categories[idx], confidence


def describe_video_segment(video_path, start_sec, end_sec):
    """
    PLACEHOLDER — Replace with Twelve Labs Pegasus analyze.

    Should:
        1. Upload/index the video (or use an already-indexed video_id)
        2. Call client.analyze(video_id=..., prompt=...) scoped to the time range
        3. Return the description string

    For now: returns a placeholder description.
    """
    # TODO: Replace with real Pegasus call
    # prompt = (
    #     f"Focus on the segment from {start_sec}s to {end_sec}s. "
    #     "Describe the people involved: their appearance, clothing, and actions. "
    #     "Describe the situation: what is happening, the setting, and any notable objects."
    # )
    # result = client.analyze(video_id=video_id, prompt=prompt)
    # return result.data

    return (
        f"[PLACEHOLDER] Segment {start_sec:.1f}s–{end_sec:.1f}s: "
        f"Two individuals in dark clothing are seen near a parked vehicle. "
        f"One appears to be acting aggressively toward the other."
    )


# ═══════════════════════════════════════════════════════════════════
# Operator 1: Detect Anomaly Windows
# ═══════════════════════════════════════════════════════════════════

class DetectAnomalyWindows(foo.Operator):
    """
    Splits each video into 10-second windows, embeds each with Marengo,
    and scores against a 'normal' baseline. Writes TemporalDetections
    for windows that exceed the anomaly threshold.
    """

    @property
    def config(self):
        return foo.OperatorConfig(
            name="detect_anomaly_windows",
            label="Detect Anomaly Windows",
            description="Find 10-second temporal windows where anomalies occur",
        )

    def resolve_input(self, ctx):
        inputs = foo.types.Object()
        inputs.str(
            "normal_label",
            label="Label to use as 'normal' baseline",
            default="Normal",
        )
        inputs.float(
            "window_sec",
            label="Window size (seconds)",
            default=10.0,
        )
        inputs.float(
            "threshold",
            label="Anomaly distance threshold (0–2)",
            default=0.5,
        )
        return foo.types.Property(inputs)

    def execute(self, ctx):
        normal_label = ctx.params.get("normal_label", "Normal")
        window_sec = ctx.params.get("window_sec", 10.0)
        threshold = ctx.params.get("threshold", 0.5)

        dataset = ctx.dataset

        # Step 1: Build normal baseline
        normal_centroid = compute_normal_baseline(dataset, normal_label)

        # Step 2: For each sample, segment → embed → score → write detections
        for sample in dataset.iter_samples(autosave=True, progress=True):
            segments = compute_segment_embeddings(sample.filepath, window_sec)

            detections = []
            for start, end, embedding in segments:
                score = cosine_distance(embedding, normal_centroid)

                if score > threshold:
                    det = fo.TemporalDetection(
                        label="anomaly",
                        support=[round(start, 2), round(end, 2)],
                        confidence=round(score, 4),
                    )
                    detections.append(det)

                # Store per-segment embedding for downstream use
                # (segment-level embeddings stored as a list on the sample)

            sample["anomaly_detections"] = fo.TemporalDetections(
                detections=detections
            )
            sample["anomaly_count"] = len(detections)

        ctx.ops.reload_dataset()


# ═══════════════════════════════════════════════════════════════════
# Operator 2: Classify Crime
# ═══════════════════════════════════════════════════════════════════

class ClassifyCrime(foo.Operator):
    """
    For each anomaly detection window, classifies the type of crime
    by comparing its Marengo embedding against text embeddings of
    the crime categories.
    """

    @property
    def config(self):
        return foo.OperatorConfig(
            name="classify_crime",
            label="Classify Crime Type",
            description="Zero-shot classify detected anomaly windows into crime categories",
        )

    def resolve_input(self, ctx):
        inputs = foo.types.Object()
        inputs.str(
            "categories",
            label="Crime categories (comma-separated)",
            default="Abuse, Assault, Arson, Arrest, Normal",
        )
        return foo.types.Property(inputs)

    def execute(self, ctx):
        raw = ctx.params.get("categories", "Abuse, Assault, Arson, Arrest, Normal")
        categories = [c.strip() for c in raw.split(",")]

        dataset = ctx.dataset

        for sample in dataset.iter_samples(autosave=True, progress=True):
            anomaly_dets = sample.get("anomaly_detections")
            if not anomaly_dets or not anomaly_dets.detections:
                continue

            classified = []
            for det in anomaly_dets.detections:
                # Use the segment embedding for classification
                # For now, placeholder uses random embedding
                segment_emb = np.random.randn(1024).tolist()
                label, confidence = classify_with_text_embeddings(
                    segment_emb, categories
                )

                classified_det = fo.TemporalDetection(
                    label=label,
                    support=det.support,
                    confidence=round(confidence, 4),
                )
                classified.append(classified_det)

            sample["crime_detections"] = fo.TemporalDetections(
                detections=classified
            )

            # Also set a sample-level label (highest confidence detection)
            if classified:
                best = max(classified, key=lambda d: d.confidence)
                sample["predicted_crime"] = fo.Classification(
                    label=best.label,
                    confidence=best.confidence,
                )

        ctx.ops.reload_dataset()


# ═══════════════════════════════════════════════════════════════════
# Operator 3: Describe Scene
# ═══════════════════════════════════════════════════════════════════

class DescribeScene(foo.Operator):
    """
    For each anomaly/crime detection window, generates a natural language
    description of the people and situation using Pegasus.
    """

    @property
    def config(self):
        return foo.OperatorConfig(
            name="describe_scene",
            label="Describe Scene",
            description="Generate text descriptions of detected anomaly windows using Pegasus",
        )

    def resolve_input(self, ctx):
        inputs = foo.types.Object()
        inputs.str(
            "source_field",
            label="Which detections to describe",
            default="crime_detections",
            description="Use 'anomaly_detections' or 'crime_detections'",
        )
        return foo.types.Property(inputs)

    def execute(self, ctx):
        source_field = ctx.params.get("source_field", "crime_detections")

        dataset = ctx.dataset

        for sample in dataset.iter_samples(autosave=True, progress=True):
            dets = sample.get(source_field)
            if not dets or not dets.detections:
                continue

            described = []
            for det in dets.detections:
                start, end = det.support
                description = describe_video_segment(
                    sample.filepath, start, end
                )

                # Create a new detection with the description attached
                described_det = fo.TemporalDetection(
                    label=det.label,
                    support=det.support,
                    confidence=det.confidence,
                )
                # Store description as a custom attribute
                described_det["description"] = description
                described.append(described_det)

            sample["scene_descriptions"] = fo.TemporalDetections(
                detections=described
            )

        ctx.ops.reload_dataset()


# ═══════════════════════════════════════════════════════════════════
# Register all operators
# ═══════════════════════════════════════════════════════════════════

def register(plugin):
    plugin.register(DetectAnomalyWindows)
    plugin.register(ClassifyCrime)
    plugin.register(DescribeScene)