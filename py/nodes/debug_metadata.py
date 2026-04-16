import logging
import json
from ..metadata_collector.metadata_processor import MetadataProcessor

logger = logging.getLogger(__name__)

class DebugMetadataLM:
    NAME = "Debug Metadata (LoraManager)"
    CATEGORY = "Lora Manager/utils"
    DESCRIPTION = "Debug node to verify metadata (with separate text outputs)"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "hidden": {
                "id": "UNIQUE_ID",
            },
        }

    # ====================== 출력 추가 ======================
    RETURN_TYPES = (
        "STRING", "STRING", "STRING", "STRING", "STRING",
        "STRING", "STRING", "STRING", "STRING", "STRING",
        "STRING"
    )
    RETURN_NAMES = (
        "metadata_text",     # 전체 JSON (기존처럼)
        "prompt",
        "negative_prompt",
        "seed",
        "steps",
        "cfg_scale",
        "sampler",
        "scheduler",
        "checkpoint",
        "loras",
        "size",
        "clip_skip"
    )

    FUNCTION = "process_metadata"

    def process_metadata(self, images, id):
        try:
            from ..metadata_collector import get_metadata
            
            metadata = get_metadata()
            metadata_dict = MetadataProcessor.to_dict(metadata, id)

            # 전체 메타데이터를 예쁜 JSON 문자열로
            metadata_text = json.dumps(metadata_dict, indent=2, ensure_ascii=False)

            # 각 항목 안전하게 추출 (없으면 "N/A" 또는 빈 문자열)
            def safe_get(key, default="N/A"):
                # metadata_dict의 최상위 또는 'meta' 안에 있을 수 있음
                if key in metadata_dict:
                    return str(metadata_dict[key])
                elif isinstance(metadata_dict.get("meta"), dict) and key in metadata_dict["meta"]:
                    return str(metadata_dict["meta"][key])
                else:
                    return default

            prompt          = safe_get("prompt")
            negative_prompt = safe_get("negative_prompt") or safe_get("negativePrompt")
            seed            = safe_get("seed")
            steps           = safe_get("steps")
            cfg_scale       = safe_get("cfg_scale") or safe_get("cfgScale")
            sampler         = safe_get("sampler")
            scheduler       = safe_get("scheduler")
            checkpoint      = safe_get("checkpoint") or safe_get("model")
            loras           = safe_get("loras")
            size            = safe_get("size") or f"{safe_get('width')}x{safe_get('height')}"
            clip_skip       = safe_get("clip_skip")

            return {
                "result": (
                    metadata_text,
                    prompt,
                    negative_prompt,
                    seed,
                    steps,
                    cfg_scale,
                    sampler,
                    scheduler,
                    checkpoint,
                    loras,
                    size,
                    clip_skip
                ),
                "ui": {
                    "metadata": [metadata_dict]
                },
            }

        except Exception as e:
            logger.error(f"Error processing metadata: {e}", exc_info=True)
            error_msg = f"Error: {str(e)}"
            
            return {
                "result": (error_msg, error_msg, error_msg, error_msg, error_msg,
                          error_msg, error_msg, error_msg, error_msg, error_msg,
                          error_msg, error_msg),
                "ui": {"metadata": [{"error": str(e)}]},
            }
