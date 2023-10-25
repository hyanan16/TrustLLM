from dataclasses import dataclass, field
from typing import Optional


@dataclass
class overallConfig:
    openai_api_key_chatgpt: Optional[str] = None
    openai_api_key_gpt_4: Optional[str] = None
    baidu_ernie_api_key: Optional[str] = None
    deepinfra_api_key: Optional[str] = None
    perspective_api_key: Optional[str] = None


@dataclass
class evalConfig:
    eval_path: Optional[str] = None
    output_path: Optional[str] = None
    eval_type: Optional[str] = None


@dataclass
class JailbreakConfig:
    data_file: Optional[str] = None
    save_file: Optional[str] = None


@dataclass
class ToxicityConfig:
    data_file: Optional[str] = None
    save_file: Optional[str] = None


