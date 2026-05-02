import os
import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types
from .backend import analyze_with_twelvelabs

try:
    from twelvelabs import TwelveLabs
except ImportError:
    TwelveLabs = None

class AnalyzeVideoClip(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="analyze_video_clip",
            label="Analyze Video Clip (Twelve Labs)",
            description="Run Twelve Labs Pegasus or Marengo on selected video clips",
            dynamic=True
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        
        # Check if API key is in environment, else ask for it
        api_key = os.environ.get("TWELVELABS_API_KEY")
        if not api_key:
            inputs.str(
                "api_key",
                label="Twelve Labs API Key",
                description="Your API key (or set TWELVELABS_API_KEY env var)",
                required=True
            )
        
        action_choices = types.RadioGroup()
        action_choices.add_choice("summarize", label="Summarize Clip (Pegasus)")
        action_choices.add_choice("find_similar", label="Find Similar Anomalies (Marengo)")
        action_choices.add_choice("zero_shot_classify", label="Zero-Shot Classify Anomalies")
        
        inputs.enum(
            "action",
            action_choices.values(),
            view=action_choices,
            label="Analysis Action",
            required=True,
            default="summarize"
        )
        
        inputs.bool(
            "occluded_investigation",
            label="Occluded Shoplifter Mode",
            description="Focus analysis on unnatural movements and hidden behaviors (e.g. carrying large boxes, obscured upper body).",
            default=False
        )
        
        # Additional settings based on action
        action_val = ctx.params.get("action")
        if action_val == "summarize":
            inputs.str(
                "prompt",
                label="Custom Prompt (Optional)",
                description="What should Pegasus summarize? E.g., 'What anomalies are present in this clip?'",
                required=False
              )
        elif action_val == "zero_shot_classify":
            inputs.str(
                "umbrella_class",
                label="Umbrella Class (Optional)",
                description="Context class mapping for the zero-shot classifier (e.g. 'Crime' or 'Anomaly').",
                default="Crime"
            )
            inputs.list(
                "classes",
                types.String(),
                label="Classification Categories",
                description="Categories to classify against (press Enter to add)",
                default=["Abuse", "Arrest", "Arson", "Assault"]
            )
            
        return types.Property(inputs)

    def execute(self, ctx):
        # Fallback if twelvelabs isn't installed
        if TwelveLabs is None:
            ctx.ops.notify("twelvelabs package is not installed. Run `pip install twelvelabs`", variant="error")
            return {"status": "error", "message": "twelvelabs not installed"}

        api_key = os.environ.get("TWELVELABS_API_KEY") or ctx.params.get("api_key")
        action = ctx.params.get("action")
        occluded_mode = ctx.params.get("occluded_investigation", False)
        
        if not api_key:
            return {"error": "No API key provided"}
            
        # Refine action/prompt if occluded shoplifter mode is enabled
        if occluded_mode:
            if action == "summarize":
                base_prompt = ctx.params.get("prompt", "")
                focused_prompt = f"{base_prompt} (Special Task: Analyze the scene for shoplifting behavior. Pay special attention to unnatural human movement, lower body mechanics, or individuals attempting to conceal themselves behind large objects like cardboard.)".strip()
                ctx.params["prompt"] = focused_prompt
            elif action == "find_similar":
                ctx.ops.notify("Occluded Mode active: Focusing search purely on lower-body movement signatures and object occlusion.", variant="info")
            elif action == "zero_shot_classify":
                umbrella = ctx.params.get("umbrella_class", "")
                umbrella_text = f" under umbrella '{umbrella}'" if umbrella else ""
                ctx.ops.notify(f"Occluded Mode active: Adjusting taxonomy to include stealth/concealment qualifiers{umbrella_text}.", variant="info")
            
        try:
            client = TwelveLabs(api_key=api_key)
            
            # Use target_view to get currently selected samples
            view = ctx.target_view()
            selected_samples = len(view)
            
            if selected_samples == 0:
                ctx.ops.notify("Please select at least one video clip first!", variant="error")
                return {"status": "error", "message": "No samples selected"}
            
            ctx.ops.notify(f"Processing {selected_samples} samples with Twelve Labs...", variant="info")
            
            # --- Execute Twelve Labs API ---
            processed_count = 0
            
            # Helper to store index ID if fallback upload is needed
            hackathon_index_id = None
            
            for sample in view:
                # For MVP use backend wrapper to upload + analyze a local file
                try:
                    res = analyze_with_twelvelabs(
                        client,
                        sample.filepath,
                        action=action,
                        prompt=ctx.params.get("prompt"),
                        umbrella=ctx.params.get("umbrella_class"),
                        classes=ctx.params.get("classes"),
                        occluded=occluded_mode,
                    )

                    if res is None:
                        ctx.ops.notify(f"No result for {sample.filepath}", variant="warning")
                        continue

                    if "summary" in res and res["summary"]:
                        sample["twelvelabs_summary"] = res["summary"]
                        sample.save()
                        processed_count += 1

                    elif "classification" in res and res["classification"]:
                        sample["twelvelabs_prediction"] = fo.Classification(label=res["classification"])
                        sample.save()
                        processed_count += 1

                    else:
                        print(f"[{sample.filepath}] Unsupported/empty response: {res}")

                except Exception as e:
                    print(f"Failed to process sample {sample.id}: {str(e)}")
            
            ctx.ops.notify(f"Successfully processed {processed_count}/{selected_samples} samples!", variant="success")
            
            # Reload dataset in UI to show the new summary/classification fields
            ctx.ops.reload_dataset()
            
            return {
                "status": "success",
                "action_performed": action,
                "samples_processed": selected_samples
            }
        except Exception as e:
            ctx.ops.notify(f"Error: {str(e)}", variant="error")
            return {"status": "error", "message": str(e)}

def register(p):
    p.register(AnalyzeVideoClip)
