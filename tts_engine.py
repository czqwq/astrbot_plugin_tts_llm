import asyncio
import os
import re
import uuid
import wave
import time
from pathlib import Path
from typing import Optional, List, Dict

import httpx
from astrbot.api import logger, AstrBotConfig

# 尝试导入 genie_tts
try:
    import genie_tts as genie
    GENIE_AVAILABLE = True
except ImportError:
    GENIE_AVAILABLE = False
    genie = None

# --- 音频参数 (必须与Genie TTS服务输出匹配) ---
BYTES_PER_SAMPLE = 2
CHANNELS = 1
SAMPLE_RATE = 32000


class TTSEngine:
    """处理所有与TTS合成相关的核心逻辑，包括文本分块、并发合成和音频合并"""

    def __init__(self, config: AstrBotConfig, http_client: httpx.AsyncClient, plugin_data_dir: Path):
        self.config = config
        self.http_client = http_client
        self.plugin_data_dir = plugin_data_dir
        self.tts_server_index = 0
        
        # GENIE 相关初始化
        self.use_local_genie = self.config.get("use_local_genie", False)
        if self.use_local_genie and GENIE_AVAILABLE:
            # 设置 GENIE 数据目录（如果配置了的话）
            genie_data_dir = self.config.get("genie_data_dir", "")
            if genie_data_dir:
                import os
                os.environ["GENIE_DATA_DIR"] = genie_data_dir
            
            # 预加载字符模型目录
            self.genie_character_models_dir = self.config.get("genie_character_models_dir", "")
            
            logger.info("GENIE 本地模型支持已启用")
        elif self.use_local_genie:
            logger.warning("GENIE 本地模型已启用但未安装 genie_tts 包，请运行 'pip install genie-tts'")
            self.use_local_genie = False
        
        # 设置临时音频目录
        self.temp_audio_dir = self.plugin_data_dir / "temp_audio"
        self.temp_audio_dir.mkdir(parents=True, exist_ok=True)
        
        # 启动清理任务
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self):
        """定期清理过期的临时音频文件"""
        while True:
            try:
                await asyncio.sleep(3600)  # 每小时检查一次
                current_time = time.time()
                expiration_time = 1800  # 30分钟过期

                if not self.temp_audio_dir.exists():
                    continue

                count = 0
                for file_path in self.temp_audio_dir.glob("*.wav"):
                    try:
                        if current_time - file_path.stat().st_mtime > expiration_time:
                            file_path.unlink()
                            count += 1
                    except Exception as e:
                        logger.warning(f"清理文件 {file_path} 失败: {e}")
                
                if count > 0:
                    logger.info(f"已清理 {count} 个过期的临时音频文件。")
                    
            except Exception as e:
                logger.error(f"清理任务发生错误: {e}")

    def _split_text_into_chunks(self, text: str, sentences_per_chunk: int) -> list[str]:
        """根据标点将文本切分为句子，再按指定数量合并成块"""
        if sentences_per_chunk <= 0:
            return [text]
            
        regex_pattern = self.config.get("sentence_split_regex", r'([。、，！？,.!?])')
        
        sentences = re.split(regex_pattern, text)
        if not sentences:
            return []

        full_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i]
            delimiter = sentences[i+1] if i+1 < len(sentences) else ""
            if sentence:
                full_sentences.append(sentence + delimiter)
        if len(sentences) % 2 == 1 and sentences[-1]:
            full_sentences.append(sentences[-1])

        chunks = []
        for i in range(0, len(full_sentences), sentences_per_chunk):
            chunk = "".join(full_sentences[i:i + sentences_per_chunk])
            chunks.append(chunk)
            
        logger.info(f"文本已切分为 {len(chunks)} 个块。")
        return chunks

    async def _merge_wav_files(self, input_paths: list[str]) -> Optional[str]:
        """以无损的方式将多个WAV文件按顺序合并为一个，并清理分块文件。"""
        if not input_paths:
            return None
        
        output_path = self.temp_audio_dir / f"{uuid.uuid4()}_merged.wav"

        try:
            with wave.open(input_paths[0], 'rb') as wf_in:
                params = wf_in.getparams()

            with wave.open(str(output_path), 'wb') as wf_out:
                wf_out.setparams(params)
                for file_path in input_paths:
                    with wave.open(file_path, 'rb') as wf_in:
                        wf_out.writeframes(wf_in.readframes(wf_in.getnframes()))
            
            logger.info(f"成功将 {len(input_paths)} 个音频文件合并到: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"合并WAV文件时出错: {e}")
            return None
        finally:
            # 无论合并成功与否，都尝试清理输入的临时分块文件
            for file_path in input_paths:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except OSError as e:
                    logger.warning(f"删除临时文件 {file_path} 失败: {e}")

    async def _attempt_synthesis_on_server(
        self, server_url: str, character_name: str, ref_audio_path: str,
        ref_audio_text: str, text: str, session_id_for_log: str, language: str = None,
    ) -> Optional[str]:
        """使用单个指定的TTS服务器尝试合成语音，并返回保存好的文件路径。"""
        logger.info(f"[{session_id_for_log}] 尝试TTS服务器: {server_url}")
        try:
            # Propagate the language parameter directly, fallback to config only if None provided
            if not language:
                language = self.config.get("tts_default_language", "jp")

            ref_payload = {
                "character_name": character_name, "audio_path": ref_audio_path, "audio_text": ref_audio_text,
                "language": language,
            }
            tts_timeout = self.config.get("tts_timeout_seconds", 120)
            response = await self.http_client.post(f"{server_url}/set_reference_audio", json=ref_payload, timeout=60)
            response.raise_for_status()

            tts_payload = {"character_name": character_name, "text": text, "split_sentence": True}
            async with self.http_client.stream("POST", f"{server_url}/tts", json=tts_payload, timeout=tts_timeout) as response_tts:
                response_tts.raise_for_status()
                output_path = self.temp_audio_dir / f"{uuid.uuid4()}.wav"
                with wave.open(str(output_path), "wb") as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(BYTES_PER_SAMPLE)
                    wf.setframerate(SAMPLE_RATE)
                    async for chunk in response_tts.aiter_bytes():
                        wf.writeframes(chunk)
                return str(output_path)
        except Exception as e:
            logger.warning(f"[{session_id_for_log}] TTS服务器 {server_url} 交互失败: {e}")
            return None

    async def _attempt_synthesis_with_genie(
        self, character_name: str, ref_audio_path: str,
        ref_audio_text: str, text: str, session_id_for_log: str, language: str = None,
    ) -> Optional[str]:
        """使用本地GENIE模型尝试合成语音，并返回保存好的文件路径。"""
        logger.info(f"[{session_id_for_log}] 尝试使用本地GENIE模型: {character_name}")
        
        if not GENIE_AVAILABLE:
            logger.error(f"[{session_id_for_log}] GENIE 未安装")
            return None
        
        try:
            # 确定语言
            if not language:
                language = self.config.get("tts_default_language", "jp")
            
            # 检查是否是预定义角色
            predefined_characters = ["mika", "37", "feibi"]  # GENIE内置角色
            if character_name.lower() in predefined_characters:
                # 对于预定义角色，直接加载
                genie.load_predefined_character(character_name.lower())
            else:
                # 对于自定义角色，尝试从指定目录加载模型
                if self.genie_character_models_dir:
                    import os
                    model_path = os.path.join(self.genie_character_models_dir, character_name)
                    if os.path.exists(model_path):
                        genie.load_character(
                            character_name=character_name,
                            onnx_model_dir=model_path,
                            language=language
                        )
                    else:
                        logger.warning(f"[{session_id_for_log}] 未找到角色 {character_name} 的模型目录: {model_path}")
                        # 尝试加载预定义角色作为后备
                        genie.load_predefined_character(character_name.lower())
                else:
                    logger.info(f"[{session_id_for_log}] 未指定模型目录，尝试加载预定义角色: {character_name}")
                    try:
                        genie.load_predefined_character(character_name.lower())
                    except Exception:
                        logger.warning(f"[{session_id_for_log}] 无法加载预定义角色 {character_name}，尝试其他方法")
                        return None
            
            # 设置参考音频
            genie.set_reference_audio(
                character_name=character_name,
                audio_path=ref_audio_path,
                audio_text=ref_audio_text
            )
            
            # 生成输出路径
            output_path = self.temp_audio_dir / f"{uuid.uuid4()}_genie.wav"
            
            # 执行TTS
            genie.tts(
                character_name=character_name,
                text=text,
                save_path=str(output_path)
            )
            
            logger.info(f"[{session_id_for_log}] GENIE TTS 合成成功: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.warning(f"[{session_id_for_log}] GENIE 本地模型合成失败: {e}")
            import traceback
            logger.debug(f"[{session_id_for_log}] GENIE 错误详情: {traceback.format_exc()}")
            return None
        
    async def _synthesis_worker(
        self, worker_id: int, task_queue: asyncio.Queue, results_list: list,
        character_name: str, ref_audio_path: str, ref_audio_text: str, session_id_for_log: str,
        language: str = None,
    ):
        """单个TTS服务器的工作进程，从队列中获取任务并处理"""
        # 如果使用本地GENIE，则使用GENIE进行所有切分块的合成
        if self.use_local_genie and GENIE_AVAILABLE:
            while not task_queue.empty():
                try:
                    task_index, chunk_text = await task_queue.get()
                except asyncio.CancelledError:
                    break
                
                log_id = f"{session_id_for_log}-chunk-{task_index+1}"
                
                audio_path = await self._attempt_synthesis_with_genie(
                    character_name=character_name, ref_audio_path=ref_audio_path,
                    ref_audio_text=ref_audio_text, text=chunk_text, session_id_for_log=log_id,
                    language=language
                )
                if audio_path:
                    logger.info(f"[Worker-{worker_id}] 成功合成块 {task_index+1} 通过本地GENIE")
                    results_list[task_index] = audio_path
                else:
                    logger.error(f"[Worker-{worker_id}] 块 {task_index+1} 通过本地GENIE合成失败。")
                    results_list[task_index] = None

                task_queue.task_done()
        else:
            # 使用远程服务器
            servers = self.config.get("tts_servers", [])
            num_servers = len(servers)
            
            while not task_queue.empty():
                try:
                    task_index, chunk_text = await task_queue.get()
                except asyncio.CancelledError:
                    break
                
                start_server_idx = worker_id % num_servers
                audio_path = None
                for i in range(num_servers):
                    server_idx = (start_server_idx + i) % num_servers
                    server_url = servers[server_idx].strip("/")
                    log_id = f"{session_id_for_log}-chunk-{task_index+1}"
                    
                    audio_path = await self._attempt_synthesis_on_server(
                        server_url, character_name, ref_audio_path, ref_audio_text, chunk_text, log_id, language=language
                    )
                    if audio_path:
                        logger.info(f"[Worker-{worker_id}] 成功合成块 {task_index+1} 于服务器 {server_url}")
                        results_list[task_index] = audio_path
                        break
                
                if not audio_path:
                    logger.error(f"[Worker-{worker_id}] 块 {task_index+1} 尝试所有服务器后仍然失败。")
                    results_list[task_index] = None

                task_queue.task_done()

    async def synthesize(
        self, character_name: str, ref_audio_path: str, ref_audio_text: str, text: str, session_id_for_log: str,
        language: str = None,
    ) -> Optional[str]:
        """执行语音合成的核心入口点，支持并发处理"""
        
        # 首先检查是否使用本地GENIE模型
        if self.use_local_genie and GENIE_AVAILABLE:
            logger.info(f"[{session_id_for_log}] 使用本地GENIE模型进行合成。")
            audio_path = await self._attempt_synthesis_with_genie(
                character_name=character_name,
                ref_audio_path=ref_audio_path,
                ref_audio_text=ref_audio_text,
                text=text,
                session_id_for_log=session_id_for_log,
                language=language,
            )
            if audio_path:
                return audio_path
            
            # 如果本地GENIE失败，回落到远程服务器
            logger.warning(f"[{session_id_for_log}] 本地GENIE合成失败，尝试远程服务器。")
        
        # 使用远程服务器
        servers = self.config.get("tts_servers", [])
        if not servers:
            logger.error(f"[{session_id_for_log}] 未配置TTS服务器，且本地GENIE不可用。")
            return None

        if self.config.get("enable_sentence_splitting", False):
            sentences_per_chunk = self.config.get("sentences_per_chunk", 2)
            text_chunks = self._split_text_into_chunks(text, sentences_per_chunk)
            
            if len(text_chunks) > 1:
                task_queue = asyncio.Queue()
                for i, chunk in enumerate(text_chunks):
                    task_queue.put_nowait((i, chunk))

                results_list = [None] * len(text_chunks)
                num_workers = min(len(servers), len(text_chunks))
                workers = [
                    asyncio.create_task(
                        self._synthesis_worker(
                            worker_id=i, task_queue=task_queue, results_list=results_list,
                            character_name=character_name, ref_audio_path=ref_audio_path,
                            ref_audio_text=ref_audio_text, session_id_for_log=session_id_for_log,
                            language=language,
                        )
                    ) for i in range(num_workers)
                ]

                logger.info(f"[{session_id_for_log}] 创建了 {num_workers} 个worker来处理 {len(text_chunks)} 个语音块...")
                await task_queue.join()
                for worker in workers:
                    worker.cancel()
                await asyncio.gather(*workers, return_exceptions=True)

                successful_paths = [path for path in results_list if path]
                
                # 如果有部分失败，或者全部失败，都需要清理已经生成的临时文件
                if len(successful_paths) != len(text_chunks):
                    logger.error(f"[{session_id_for_log}] 部分或全部语音块合成失败，正在清理临时文件。")
                    for path in successful_paths:
                         try:
                            if os.path.exists(path):
                                os.remove(path)
                         except Exception as e:
                             logger.warning(f"清理残留文件 {path} 失败: {e}")
                    return None
                
                return successful_paths[0] if len(successful_paths) == 1 else await self._merge_wav_files(successful_paths)

        # 如果不切分，则使用轮询逻辑
        logger.info(f"[{session_id_for_log}] 使用单块模式进行合成。")
        start_index = self.tts_server_index
        for i in range(len(servers)):
            current_index = (start_index + i) % len(servers)
            server_url = servers[current_index].strip("/")
            
            if i == 0:
                self.tts_server_index = (start_index + 1) % len(servers)

            audio_path = await self._attempt_synthesis_on_server(
                server_url=server_url, character_name=character_name,
                ref_audio_path=ref_audio_path, ref_audio_text=ref_audio_text,
                text=text, session_id_for_log=session_id_for_log, language=language,
            )
            if audio_path:
                return audio_path

        logger.error(f"[{session_id_for_log}] 尝试所有TTS服务器后合成失败。")
        return None