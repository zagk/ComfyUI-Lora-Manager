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

    # ====================== 출력 정의 (13개) ======================
    RETURN_TYPES = (
        "STRING", "STRING", "STRING",     # metadata_text, prompt, negative_prompt
        "STRING",                         # ← 새로 추가: no_prompt (prompt 제외 버전)
        "STRING", "STRING", "STRING", "STRING", "STRING",
        "STRING", "STRING", "STRING", "STRING"
    )

    RETURN_NAMES = (
        "metadata_text",      # 0: 전체 JSON
        "prompt",             # 1
        "negative_prompt",    # 2
        "no_prompt",          # 3: ← 새로 추가 (prompt, negative_prompt 제외한 JSON)
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
            
            # 1. 전체 메타데이터 JSON
            metadata_text = json.dumps(metadata_dict, indent=2, ensure_ascii=False)

            # 2. prompt와 negative_prompt 제외한 버전 만들기
            metadata_no_prompt = metadata_dict.copy()  # 원본 복사
            
            # prompt 관련 키들 제거 (가능한 모든 키 이름 대응)
            keys_to_remove = ["prompt", "negative_prompt", "negativePrompt", "positive_prompt"]
            for key in keys_to_remove:
                metadata_no_prompt.pop(key, None)
                if isinstance(metadata_no_prompt.get("meta"), dict):
                    metadata_no_prompt["meta"].pop(key, None)

            no_prompt_text = json.dumps(metadata_no_prompt, indent=2, ensure_ascii=False)

            # 안전하게 값 가져오는 함수
            def safe_get(key, default="N/A"):
                if key in metadata_dict:
                    return str(metadata_dict[key])
                elif isinstance(metadata_dict.get("meta"), dict) and key in metadata_dict["meta"]:
                    return str(metadata_dict["meta"][key])
                else:
                    return default

            prompt = safe_get("prompt")
            negative_prompt = safe_get("negative_prompt") or safe_get("negativePrompt")
            seed = safe_get("seed")
            steps = safe_get("steps")
            cfg_scale = safe_get("cfg_scale") or safe_get("cfgScale")
            sampler = safe_get("sampler")
            scheduler = safe_get("scheduler")
            checkpoint = safe_get("checkpoint") or safe_get("model")
            loras = safe_get("loras")
            size = safe_get("size") or f"{safe_get('width')}x{safe_get('height')}"
            clip_skip = safe_get("clip_skip")

            return {
                "result": (
                    metadata_text,      # 0
                    prompt,             # 1
                    negative_prompt,    # 2
                    no_prompt_text,     # 3 ← 새로 추가
                    seed,               # 4
                    steps,              # 5
                    cfg_scale,          # 6
                    sampler,            # 7
                    scheduler,          # 8
                    checkpoint,         # 9
                    loras,              # 10
                    size,               # 11
                    clip_skip           # 12
                ),
                "ui": {
                    "metadata": [metadata_dict]
                },
            }

        except Exception as e:
            logger.error(f"Error processing metadata: {e}", exc_info=True)
            error_msg = f"Error: {str(e)}"
           
            # 13개 모두 에러 메시지로 반환
            return {
                "result": (error_msg,) * 13,
                "ui": {"metadata": [{"error": str(e)}]},
            }
