from typing_extensions import override
from comfy_api.latest import ComfyExtension

from . import nodes

class Extension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [
            nodes.IrodoriTTS_v2ModelLoader, 
            nodes.IrodoriTTS_v2ReferenceAudio, 
            nodes.IrodoriTTS_v2AdvancedCFG, 
            nodes.IrodoriTTS_v2RescaleConfig, 
            nodes.IrodoriTTS_v2Sampler, 
            nodes.IrodoriTTS_v2DialogueTTS,
            nodes.IrodoriTTS_v2EmojiSelector, 
        ]


async def comfy_entrypoint():
    return Extension()

WEB_DIRECTORY = "./web"


