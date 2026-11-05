# litert-community catalog snapshot (live HF API, 2026-11-05)

Source: `GET https://huggingface.co/api/models?author=litert-community&limit=1000&full=true`
(fetched live from this session; 443 public repos). This snapshot **supersedes** the
"no LiteRT TTS exists" statement in `research-litert-ecosystem.md`, which was written
while huggingface.co was blocked from the container.

## Repos by task

| pipeline_tag | repos |
|---|---|
| image-classification | 119 |
| text-generation | 116 |
| image-text-to-text | 29 |
| automatic-speech-recognition | 26 |
| image-segmentation | 21 |
| (none) | 20 |
| text-classification | 13 |
| text-to-speech | 13 |
| image-to-image | 12 |
| feature-extraction | 10 |
| object-detection | 9 |
| keypoint-detection | 8 |
| depth-estimation | 6 |
| token-classification | 6 |
| audio-to-audio | 5 |
| image-feature-extraction | 5 |
| sentence-similarity | 4 |
| text-to-image | 4 |
| audio-classification | 4 |
| any-to-any | 2 |
| text-ranking | 2 |
| question-answering | 1 |
| mask-generation | 1 |
| video-classification | 1 |
| translation | 1 |
| text-to-video | 1 |
| image-to-text | 1 |
| voice-activity-detection | 1 |
| zero-shot-classification | 1 |
| audio-text-to-text | 1 |

## ASR (speech → text)

| repo | downloads | formats | language tags |
|---|---|---|---|
| [litert-community/parakeet-tdt-0.6b-v3](https://huggingface.co/litert-community/parakeet-tdt-0.6b-v3) | 12818 | tflite, litertlm | bg hr cs da nl en et fi … |
| [litert-community/whisper-tiny](https://huggingface.co/litert-community/whisper-tiny) | 5683 | tflite, litertlm | en zh de es ru ko fr ja … |
| [litert-community/moonshine-tiny](https://huggingface.co/litert-community/moonshine-tiny) | 4081 | tflite, litertlm | en |
| [litert-community/whisper-base](https://huggingface.co/litert-community/whisper-base) | 1607 | tflite |  |
| [litert-community/Qwen3-ASR-0.6B](https://huggingface.co/litert-community/Qwen3-ASR-0.6B) | 1296 | tflite, litertlm | zh en ar de fr es pt id … |
| [litert-community/whisper-acft](https://huggingface.co/litert-community/whisper-acft) | 1249 | tflite |  |
| [litert-community/parakeet-ctc-0.6b](https://huggingface.co/litert-community/parakeet-ctc-0.6b) | 1188 | tflite, litertlm | en |
| [litert-community/Qwen3-ASR-1.7B](https://huggingface.co/litert-community/Qwen3-ASR-1.7B) | 1088 | litertlm | zh en ar de fr es pt id … |
| [litert-community/Speaker-Diarization-LiteRT](https://huggingface.co/litert-community/Speaker-Diarization-LiteRT) | 755 | tflite |  |
| [litert-community/whisper-medium](https://huggingface.co/litert-community/whisper-medium) | 518 | tflite |  |
| [litert-community/parakeet-redux](https://huggingface.co/litert-community/parakeet-redux) | 390 | tflite | en de fr es it pt ru uk … |
| [litert-community/Fun-ASR-Nano-2512](https://huggingface.co/litert-community/Fun-ASR-Nano-2512) | 385 | litertlm | zh en ja |
| [litert-community/VibeVoice-ASR-Streaming-1.5B](https://huggingface.co/litert-community/VibeVoice-ASR-Streaming-1.5B) | 377 | litertlm | en zh es pt de ja ko fr … |
| [litert-community/VibeVoice-ASR-BitNet](https://huggingface.co/litert-community/VibeVoice-ASR-BitNet) | 308 | litertlm | en zh fr it ko pt vi |
| [litert-community/whisper-large-v3-turbo](https://huggingface.co/litert-community/whisper-large-v3-turbo) | 261 | tflite |  |
| [litert-community/parakeet-ultra](https://huggingface.co/litert-community/parakeet-ultra) | 216 | tflite | en de fr es it pt ru uk … |
| [litert-community/Zipformer-medium-CR-CTC-LiteRT](https://huggingface.co/litert-community/Zipformer-medium-CR-CTC-LiteRT) | 204 | tflite | en |
| [litert-community/parakeet-tdt_ctc-0.6b-ja](https://huggingface.co/litert-community/parakeet-tdt_ctc-0.6b-ja) | 201 | tflite | ja |
| [litert-community/SenseVoiceSmall-ko-zeroth](https://huggingface.co/litert-community/SenseVoiceSmall-ko-zeroth) | 191 | tflite | ko en |
| [litert-community/SenseVoiceSmall](https://huggingface.co/litert-community/SenseVoiceSmall) | 122 | tflite | ko en zh ja |
| [litert-community/wav2vec2-base-960h-LiteRT](https://huggingface.co/litert-community/wav2vec2-base-960h-LiteRT) | 91 | tflite | en |
| [litert-community/granite-speech-5.0-470m-turboctc](https://huggingface.co/litert-community/granite-speech-5.0-470m-turboctc) | 85 | tflite | en |
| [litert-community/japanese-zipformer-base-LiteRT](https://huggingface.co/litert-community/japanese-zipformer-base-LiteRT) | 74 | tflite | ja |
| [litert-community/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/litert-community/nemotron-3.5-asr-streaming-0.6b) | 72 | tflite | en es de fr it ar ja ko … |
| [litert-community/whisper-large-v3](https://huggingface.co/litert-community/whisper-large-v3) | 54 | tflite |  |
| [litert-community/Breeze-ASR-26-LiteRT](https://huggingface.co/litert-community/Breeze-ASR-26-LiteRT) | 9 | tflite | zh |

## TTS (text → speech)

| repo | downloads | formats | language tags |
|---|---|---|---|
| [litert-community/Kokoro-82M](https://huggingface.co/litert-community/Kokoro-82M) | 7861 | tflite, litertlm | en es fr it pt hi zh ja |
| [litert-community/sopro-v2-turbo](https://huggingface.co/litert-community/sopro-v2-turbo) | 4717 | tflite | en pt fr de |
| [litert-community/Matcha-TTS](https://huggingface.co/litert-community/Matcha-TTS) | 2825 | tflite | en |
| [litert-community/Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/litert-community/Qwen3-TTS-12Hz-0.6B-Base) | 987 | tflite, litertlm | en zh ja ko es fr de it … |
| [litert-community/kitten-tts-nano-0.8](https://huggingface.co/litert-community/kitten-tts-nano-0.8) | 860 | tflite | en |
| [litert-community/Audio8-TTS-Preview-0.6b](https://huggingface.co/litert-community/Audio8-TTS-Preview-0.6b) | 500 | tflite | en zh ja ko de fr es it … |
| [litert-community/Inflect-Nano-v2](https://huggingface.co/litert-community/Inflect-Nano-v2) | 353 | tflite | en |
| [litert-community/EMA-Lightning-LiteRT](https://huggingface.co/litert-community/EMA-Lightning-LiteRT) | 325 | tflite | tr |
| [litert-community/Ming-3.0-tts-0.5B](https://huggingface.co/litert-community/Ming-3.0-tts-0.5B) | 291 | tflite | en zh ja ko de fr es |
| [litert-community/Audio8-TTS-Realistic-0.6b](https://huggingface.co/litert-community/Audio8-TTS-Realistic-0.6b) | 146 | tflite | zh en ja ko de fr es it … |
| [litert-community/Kokoro-G2P-en-US](https://huggingface.co/litert-community/Kokoro-G2P-en-US) | 40 | tflite |  |
| [litert-community/phonon-31m-en](https://huggingface.co/litert-community/phonon-31m-en) | 0 | litertlm | en |
| [litert-community/pocket-tts](https://huggingface.co/litert-community/pocket-tts) | 0 | litertlm | en fr de es it pt nl |

## Top 25 LLMs (text-generation) by downloads

| repo | downloads | formats |
|---|---|---|
| [litert-community/Qwen2.5-1.5B-Instruct](https://huggingface.co/litert-community/Qwen2.5-1.5B-Instruct) | 90177 | tflite, litertlm, task |
| [litert-community/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/litert-community/DeepSeek-R1-Distill-Qwen-1.5B) | 68652 | tflite, litertlm, task |
| [litert-community/Qwen3-0.6B](https://huggingface.co/litert-community/Qwen3-0.6B) | 59803 | litertlm |
| [litert-community/Phi-4-mini-instruct](https://huggingface.co/litert-community/Phi-4-mini-instruct) | 20292 | tflite, litertlm, task |
| [litert-community/Qwen3-1.7B](https://huggingface.co/litert-community/Qwen3-1.7B) | 18227 | litertlm |
| [litert-community/Qwen2.5-0.5B-Instruct](https://huggingface.co/litert-community/Qwen2.5-0.5B-Instruct) | 12064 | tflite, task |
| [litert-community/Gemma3-1B-IT](https://huggingface.co/litert-community/Gemma3-1B-IT) | 11327 | litertlm, task |
| [litert-community/Qwen3-4B](https://huggingface.co/litert-community/Qwen3-4B) | 11070 | litertlm |
| [litert-community/Ternary-Bonsai-1.7B](https://huggingface.co/litert-community/Ternary-Bonsai-1.7B) | 8352 | litertlm |
| [litert-community/SmolLM2-360M-Instruct](https://huggingface.co/litert-community/SmolLM2-360M-Instruct) | 6962 | litertlm |
| [litert-community/LFM2.5-1.2B-Instruct](https://huggingface.co/litert-community/LFM2.5-1.2B-Instruct) | 5812 | litertlm |
| [litert-community/Qwen3.5-4B](https://huggingface.co/litert-community/Qwen3.5-4B) | 5279 | litertlm |
| [litert-community/Qwen3.5-2B](https://huggingface.co/litert-community/Qwen3.5-2B) | 5201 | litertlm |
| [litert-community/Qwen3-0.6B-int4](https://huggingface.co/litert-community/Qwen3-0.6B-int4) | 4473 | litertlm |
| [litert-community/Qwen3.5-0.8B](https://huggingface.co/litert-community/Qwen3.5-0.8B) | 4305 | litertlm |
| [litert-community/Qwen2.5-Coder-3B-Instruct](https://huggingface.co/litert-community/Qwen2.5-Coder-3B-Instruct) | 3520 | litertlm |
| [litert-community/Qwen3-4B-Instruct-2507](https://huggingface.co/litert-community/Qwen3-4B-Instruct-2507) | 3427 | litertlm |
| [litert-community/SmolLM2-135M-Instruct](https://huggingface.co/litert-community/SmolLM2-135M-Instruct) | 3318 | litertlm |
| [litert-community/SmolLM3-3B](https://huggingface.co/litert-community/SmolLM3-3B) | 3025 | litertlm |
| [litert-community/Qwen3-14B](https://huggingface.co/litert-community/Qwen3-14B) | 2713 | litertlm |
| [litert-community/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/litert-community/TinyLlama-1.1B-Chat-v1.0) | 2299 | tflite, task |
| [litert-community/Qwen3-8B](https://huggingface.co/litert-community/Qwen3-8B) | 2273 | litertlm |
| [litert-community/Ministral-3-3B-Instruct-2512](https://huggingface.co/litert-community/Ministral-3-3B-Instruct-2512) | 2252 | litertlm |
| [litert-community/Gemma3-1B-IT-Tensor-NPU](https://huggingface.co/litert-community/Gemma3-1B-IT-Tensor-NPU) | 2179 | litertlm |
| [litert-community/DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/litert-community/DeepSeek-R1-Distill-Qwen-7B) | 1935 | litertlm |
