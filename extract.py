import os
import argparse
import subprocess
import glob
from pathlib import Path
from pydub import AudioSegment
from groq import Groq
from dotenv import load_dotenv
import math
import concurrent.futures
from tqdm import tqdm
import time
import re
import sys
from datetime import datetime
import json

# 加载环境变量
load_dotenv()

# 初始化 Groq 客户端
api_key = os.getenv("GROQ_API_KEY")
client = None
if api_key:
    client = Groq(api_key=api_key)

def get_audio_duration(file_path):
    """获取音频文件时长（秒）"""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", file_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL # 彻底静默 ffprobe 报错
    )
    try:
        return float(result.stdout)
    except:
        return 0

def split_audio(file_path, chunk_length_ms=10*60*1000):
    """
    将音频分割成小块，每块默认为10分钟
    返回临时文件路径列表
    """
    tqdm.write(f"正在分割音频: {os.path.basename(file_path)}...")
    audio = AudioSegment.from_mp3(file_path)
    chunks = []
    
    # 25MB limit roughly translates to ~20 mins at 128kbps, but being safe with 10 mins
    # Groq limit is file size based (25MB). 
    
    total_length = len(audio)
    num_chunks = math.ceil(total_length / chunk_length_ms)
    
    temp_dir = Path(file_path).parent / ".temp_chunks"
    temp_dir.mkdir(exist_ok=True)
    
    files = []
    for i in range(num_chunks):
        start = i * chunk_length_ms
        end = min((i + 1) * chunk_length_ms, total_length)
        chunk = audio[start:end]
        
        chunk_name = temp_dir / f"{Path(file_path).stem}_chunk_{i}.mp3"
        chunk.export(chunk_name, format="mp3")
        files.append(str(chunk_name))
        
    return files, temp_dir

    try:
        with open(file_path, "rb") as file:
            # 使用 verbose_json 以获取分段信息，从而优化排版
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(file_path), file.read()),
                model="whisper-large-v3",
                response_format="verbose_json", 
                language="zh" # 强制中文
            )
        
        # 尝试提取分段文本以保留段落感
        if hasattr(transcription, 'segments'):
            segments = [seg['text'].strip() for seg in transcription.segments]
            return "\n".join(segments)
        else:
            return transcription.text
            
    except Exception as e:
        error_msg = str(e)
        # 检查是否是频率限制 (Rate Limit - 429)
        if "Rate limit reached" in error_msg or "429" in error_msg:
            wait_hint = "几分钟"
            match = re.search(r"try again in (\d+m)?(\d+s)?", error_msg)
            if match:
                wait_hint = f"{match.group(1) or ''}{match.group(2) or ''}"
            
            tqdm.write("\n" + "!"*50)
            tqdm.write(f"🛑 Groq API 额度已达上限 (Rate Limit)。")
            tqdm.write(f"由于你当前的 API Key 是免费层级，请等待约 {wait_hint} 后再试。")
            tqdm.write("程序已中断退出。")
            tqdm.write("!"*50 + "\n")
            sys.exit(1) # 按用户要求，直接中断退出
        
        # 其他异常直接抛出，由上层捕获
        raise RuntimeError(f"Transcription failed for {os.path.basename(file_path)}: {str(e)}")

def transcribe_audio_file(mp3_path):
    """转录完整音频文件（处理分割和合并）"""
    if not client:
        tqdm.write("警告: 未设置 GROQ_API_KEY，跳过转录")
        return None
    
    tqdm.write(f"开始转录: {os.path.basename(mp3_path)}")
    
    # 检查文件大小，如果小于 20MB 直接转录
    file_size_mb = os.path.getsize(mp3_path) / (1000 * 1000)
    if file_size_mb < 20:
        return transcribe_chunk(mp3_path)
    
    # 大文件分割处理
    files, temp_dir = split_audio(mp3_path)
    full_text = []
    
    # 使用线程池并发转录
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # 提交所有任务
            future_to_file = {executor.submit(transcribe_chunk, f): f for f in files}
            
            # 使用 tqdm 显示进度
            results = []
            # 按顺序收集结果需要保持 index 顺序，但 as_completed 是乱序的。
            # 这里为了简单 robust，我们直接 map，但 map 会在遇到异常时通过 iterator 抛出
            for text in tqdm(executor.map(transcribe_chunk, files), total=len(files), desc="转录进度", leave=False):
                results.append(text)
                
        full_text = "\n\n".join(results) # 块之间双换行分隔
        
        # 清理临时文件
        for f in files:
            os.remove(f)
        os.rmdir(temp_dir)
        
        return full_text
        
    except Exception as e:
        tqdm.write(f"❌ 文件 {os.path.basename(mp3_path)} 转录过程中出错: {e}")
        # 出错时不返回任何文本，避免生成垃圾文件
        # 清理残余
        for f in files:
            if os.path.exists(f): os.remove(f)
        if os.path.exists(temp_dir): os.rmdir(temp_dir)
        return None

def verify_source_integrity(file_path):
    """快速检查源视频文件完整性"""
    # 1. 检查是否存在基本的流信息
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_name", "-of", "csv=p=0", file_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )
    if not result.stdout.strip():
        return False, "找不到音频流"

    # 2. 快速抽样扫描 (检查开头和中间)
    # 使用 -f null 测试前 10 秒
    check_cmd = [
        "ffmpeg", "-v", "error", "-ss", "0", "-i", file_path, 
        "-t", "10", "-f", "null", "-", "-threads", "1"
    ]
    check_res = subprocess.run(check_cmd, stderr=subprocess.PIPE)
    if check_res.returncode != 0:
        return False, "视频开头数据损坏"
        
    return True, "正常"

def check_duration_match(source_path, target_path, threshold_sec=10):
    """验证提取后的音频时长是否与视频时长匹配"""
    source_dur = get_audio_duration(source_path)
    target_dur = get_audio_duration(target_path)
    
    if source_dur == 0 or target_dur == 0:
        return True, 0, 0 # 无法获取时长时跳过校验
        
    diff = abs(source_dur - target_dur)
    is_match = diff <= threshold_sec or (diff / source_dur) < 0.02 # 误差在10秒或2%以内
    return is_match, source_dur, target_dur

def extract_mp3(mp4_path):
    """使用 FFmpeg 提取 MP3，带时长校验和深度修复逻辑"""
    mp3_path = str(Path(mp4_path).with_suffix('.mp3'))

    # 检查原始MP3文件是否存在
    if os.path.exists(mp3_path):
        meta = get_file_metadata(mp3_path)
        tqdm.write(f"⏭️ MP3 已存在，跳过: {os.path.basename(mp3_path)} ({meta['size']}) [{meta['time']}]")
        return mp3_path

    # 检查是否存在分卷文件（如 file_1.mp3, file_2.mp3）
    base_name = Path(mp4_path).with_suffix('')
    split_files = sorted(glob.glob(f"{base_name}_*.mp3"))
    if split_files:
        tqdm.write(f"⏭️ 检测到已存在的分卷MP3文件，跳过:")
        for f in split_files:
            meta = get_file_metadata(f)
            tqdm.write(f"   -> {meta['name']} ({meta['size']}) [{meta['time']}]")
        return "|".join(split_files)
        
    tqdm.write(f"正在进行源文件体检: {os.path.basename(mp4_path)}")
    is_healthy, reason = verify_source_integrity(mp4_path)
    if not is_healthy:
        tqdm.write(f"⚠️ 源文件疑似异常: {reason}")
        tqdm.write("将尝试使用强制提取模式，请在完成后务必核对时长。")

    tqdm.write(f"正在提取音频: {os.path.basename(mp4_path)}")
    source_dur = get_audio_duration(mp4_path)
    
    # 策略 1: 增强型标准提取
    cmd_1 = [
        "ffmpeg", "-analyzeduration", "100M", "-probesize", "100M",
        "-i", mp4_path, "-vn", "-dn",
        "-acodec", "libmp3lame", "-q:a", "2", 
        "-fflags", "+genpts+discardcorrupt",
        "-y", "-hide_banner", "-loglevel", "error", mp3_path
    ]
    
    # 策略 2: 深度修复模式 (处理截断问题)
    cmd_deep = [
        "ffmpeg", "-analyzeduration", "200M", "-probesize", "200M",
        "-fflags", "+genpts+igndts+discardcorrupt", # 忽略不连续的时间戳
        "-err_detect", "ignore_err",
        "-i", mp4_path, "-vn", "-dn",
        "-acodec", "libmp3lame", "-q:a", "4",
        "-af", "aresample=async=1", # 强制音频同步
        "-max_muxing_queue_size", "2048",
        "-y", "-hide_banner", "-loglevel", "error", mp3_path
    ]

    try:
        # 尝试策略 1
        subprocess.run(cmd_1, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # 时长校验
        is_match, s_dur, t_dur = check_duration_match(mp4_path, mp3_path)
        if not is_match:
            tqdm.write(f"⚠️ 检测到音频截断 (视频 {int(s_dur)}s -> 音频 {int(t_dur)}s)，启动深度修复...")
            subprocess.run(cmd_deep, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # 二次校验
            is_match, s_dur, t_dur = check_duration_match(mp4_path, mp3_path)
            if not is_match:
                tqdm.write(f"❌ 深度修复后时长仍不匹配 (缺失 {int(s_dur - t_dur)}秒)")
                return "TRUNCATED:" + mp3_path
                
        # 检查文件大小并自动切割 (199MB 限制)
        final_files = check_and_split_by_size(mp3_path)
        for f in final_files:
            meta = get_file_metadata(f)
            tqdm.write(f"✅ 提取完成: {meta['name']} ({meta['size']}) [{meta['time']}]")
        return "|".join(final_files)
        
    except subprocess.CalledProcessError:
        tqdm.write(f"⚠️ 提取失败，尝试深度修复模式: {os.path.basename(mp4_path)}")
        try:
            subprocess.run(cmd_deep, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            final_files = check_and_split_by_size(mp3_path)
            return "|".join(final_files)
        except subprocess.CalledProcessError as e:
            tqdm.write(f"❌ 提取失败 (跳过该文件): {e}")
            return None

def check_and_split_by_size(mp3_path, max_size_mb=199):
    """如果文件超过大小限制，则进行切割"""
    if not os.path.exists(mp3_path):
        return []
        
    file_size_mb = os.path.getsize(mp3_path) / (1000 * 1000)
    if file_size_mb <= max_size_mb:
        return [mp3_path]
        
    tqdm.write(f"📦 文件过大 ({int(file_size_mb)}MB)，正在自动分卷...")
    duration = get_audio_duration(mp3_path)
    if duration <= 0:
        return [mp3_path]
        
    # 计算需要分多少份 (保守估计)
    num_parts = math.ceil(file_size_mb / (max_size_mb - 5))
    part_duration = duration / num_parts
    
    parts = []
    base_name = Path(mp3_path).with_suffix('')
    
    for i in range(num_parts):
        start_time = i * part_duration
        part_path = f"{base_name}_{i+1}.mp3"
        
        cmd = [
            "ffmpeg", "-i", mp3_path, "-ss", str(start_time), "-t", str(part_duration),
            "-acodec", "copy", "-y", "-hide_banner", "-loglevel", "error", part_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        parts.append(part_path)
        
    # 删除原大文件以节省空间
    os.remove(mp3_path)
    tqdm.write(f"✅ 已切分为 {num_parts} 个文件")
    return parts

def is_valid_text(text):
    """检查文本是否包含错误信息或过短"""
    if not text or len(text) < 10:
        return False
    
    error_markers = [
        "Error code:", 
        "Rate limit reached", 
        "403 Forbidden", 
        "429 Too Many Requests"
    ]
    
    # 检查前 200 个字符即可 (通常错误在开头)
    sample = text[:200]
    for marker in error_markers:
        if marker in sample:
            return False
            
    return True

def should_skip_text_generation(txt_path):
    """智能跳过检测：如果文件存在且内容有效，则跳过"""
    if not os.path.exists(txt_path):
        return False
        
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        if is_valid_text(content):
            return True
        else:
            tqdm.write(f"⚠️ 发现无效的旧文本文件 (包含错误信息)，准备重新生成: {os.path.basename(txt_path)}")
            return False
    except:
        return False

def save_text(text, original_path):
    if not is_valid_text(text):
        tqdm.write(f"❌ 生成的文本无效 (包含错误信息)，不予保存: {os.path.basename(original_path)}")
        return

    txt_path = str(Path(original_path).with_suffix('.txt'))
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    tqdm.write(f"文本已保存: {os.path.basename(txt_path)}")

def get_file_metadata(path):
    """获取文件的基本元数据：大小和最后修改时间"""
    p = Path(path)
    if not p.exists():
        return None
    size_bytes = p.stat().st_size
    # 格式化大小
    if size_bytes > 1024 * 1024:
        size_str = f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        size_str = f"{size_bytes / 1024:.2f} KB"
    
    timestamp = datetime.fromtimestamp(p.stat().st_mtime).strftime("%H:%M:%S")
    return {"name": p.name, "size": size_str, "time": timestamp}

def save_execution_log(results, root_dir, mode, start_time):
    """保存执行日志到本地文本文件"""
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"log_{timestamp}.txt"
    end_time = datetime.now()
    duration = end_time - start_time
    
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write(f"执行日志 - {timestamp}\n")
        f.write("="*80 + "\n")
        f.write(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总耗时:   {duration}\n")
        f.write(f"处理目录: {root_dir}\n")
        f.write(f"运行模式: {mode}\n")
        f.write("-" * 80 + "\n")
        
        counts = {"✅ 成功": 0, "❌ 失败": 0, "⚠️ 截断": 0, "⏭️ 跳过": 0, "❌ 出错": 0}
        for res in results:
            status = res['status']
            counts[status] = counts.get(status, 0) + 1
            f.write(f"[{status}] {res['file']}\n")
            if res.get('outputs'):
                for out in res['outputs']:
                    f.write(f"    -> {out['name']} ({out['size']}) [{out['time']}]\n")
            elif res.get('note'):
                f.write(f"    备注: {res['note']}\n")
            f.write("\n")
            
        f.write("-" * 80 + "\n")
        f.write("处理汇总:\n")
        for k, v in counts.items():
            if v > 0:
                f.write(f"  {k}: {v} 个\n")
        f.write("="*80 + "\n")
        
    return log_file

def process_directory(root_dir, mode="all", workers=1):
    """
    mode: 
    - "all": MP4 -> MP3 + TXT
    - "mp3_to_txt": MP3 -> TXT
    - "mp4_to_mp3": MP4 -> MP3 (仅提取音频)
    """
    start_time = datetime.now()
    root_path = Path(root_dir)
    
    if mode in ["all", "mp4_to_mp3"]:
        # 查找所有 MP4
        # glob recursive search
        mp4_files = list(root_path.rglob("*.mp4"))
        # 过滤掉以 ._ 开头的隐藏文件
        mp4_files = [f for f in mp4_files if not f.name.startswith("._")]
        
        results = []
        pbar_total = tqdm(mp4_files, desc="总任务进度", position=1, leave=True)
        
        # 定义单文件处理逻辑以便并行调用
        def process_single_mp4(mp4):
            try:
                # 在多线程下，我们使用 tqdm.write 来避免进度条冲突
                mp3_res = extract_mp3(str(mp4))
                if mp3_res:
                    is_truncated = mp3_res.startswith("TRUNCATED:")
                    mp3_parts = mp3_res.replace("TRUNCATED:", "").split("|")
                    
                    if mode == "mp4_to_mp3":
                        outputs = [get_file_metadata(p) for p in mp3_parts]
                        status = "⚠️ 截断" if is_truncated else "✅ 成功"
                        note = f"切分{len(mp3_parts)}份" if len(mp3_parts) > 1 else ("音频不完整" if is_truncated else "仅提取音频")
                        return {"file": mp4.name, "status": status, "note": note, "outputs": outputs}
                        
                    txt_path = str(Path(mp4).with_suffix('.txt'))
                    if should_skip_text_generation(txt_path):
                        return {"file": mp4.name, "status": "⏭️ 跳过", "note": "文本已存在"}

                    combined_text = []
                    for part in mp3_parts:
                        text = transcribe_audio_file(part)
                        if text: combined_text.append(text)
                    
                    if combined_text:
                        save_text("\n\n".join(combined_text), mp4)
                        outputs = [get_file_metadata(p) for p in mp3_parts]
                        outputs.append(get_file_metadata(txt_path))
                        return {
                            "file": mp4.name, 
                            "status": "✅ 成功", 
                            "note": f"含{len(mp3_parts)}分卷" if len(mp3_parts) > 1 else "",
                            "outputs": outputs
                        }
                    return {"file": mp4.name, "status": "❌ 失败", "note": "转录任务未完成"}
                return {"file": mp4.name, "status": "❌ 失败", "note": "提取音频失败"}
            except Exception as e:
                return {"file": mp4.name, "status": "❌ 出错", "note": str(e)}
            finally:
                pbar_total.update(1)

        if workers is None:
            if len(mp4_files) > 20:
                workers = 4 if mode == "mp4_to_mp3" else 2
                tqdm.write(f"🚀 检测到待处理文件较多 ({len(mp4_files)}), 已自动开启 {workers} 线程加速")
            else:
                workers = 1

        # 根据 workers 参数决定是否并行
        if workers > 1:
            tqdm.write(f"🚀 开启多线程模式: {workers} 个并行任务")
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                # 提交所有任务并等待
                results = list(executor.map(process_single_mp4, mp4_files))
        else:
            for mp4 in mp4_files:
                pbar_total.set_description(f"任务: {mp4.name}")
                res = process_single_mp4(mp4)
                results.append(res)
        
        pbar_total.close()

    elif mode == "mp3_to_txt":
        mp3_files = list(root_path.rglob("*.mp3"))
        # 过滤掉以 ._ 开头的隐藏文件
        mp3_files = [f for f in mp3_files if not f.name.startswith("._")]
        
        tqdm.write(f"找到 {len(mp3_files)} 个 MP3 文件")
        
        results = []
        
        for mp3 in tqdm(mp3_files, desc="处理文件"):
            try:
                # 检查 TXT 是否已存在且有效
                txt_path = str(Path(mp3).with_suffix('.txt'))
                if should_skip_text_generation(txt_path):
                    results.append({"file": mp3.name, "status": "⏭️ 跳过", "note": "文本已存在"})
                    continue

                text = transcribe_audio_file(str(mp3))
                if text:
                    save_text(text, mp3)
                    results.append({"file": mp3.name, "status": "✅ 成功", "note": ""})
                else:
                    results.append({"file": mp3.name, "status": "❌ 失败", "note": "转录中断"})
            except Exception as e:
                tqdm.write(f"处理出错: {e}")
                results.append({"file": mp3.name, "status": "❌ 出错", "note": str(e)})
                
    # 保存日志
    log_path = save_execution_log(results, root_dir, mode, start_time)

    # 输出总结报告
    print("\n" + "="*50)
    print("处理结果汇总")
    print("="*50)
    for res in results:
        note = f" ({res['note']})" if res['note'] else ""
        print(f"{res['status']} : {res['file']}{note}")
    print("="*50)
    print(f"日志路径: {log_path}")
    print("="*50 + "\n")

def main():
    parser = argparse.ArgumentParser(description="MP4音频提取与Groq极速转录工具")
    parser.add_argument("dir", help="要处理的根目录路径")
    parser.add_argument("--mode", choices=["all", "mp3_to_txt", "mp4_to_mp3"], default="mp4_to_mp3", 
                        help="模式: 'mp4_to_mp3' (默认, 仅提取音频), 'all' (MP4转MP3+文本) 或 'mp3_to_txt' (仅MP3转文本)")
    parser.add_argument("--key", help="Groq API Key (也可以通过环境变量 GROQ_API_KEY 设置)")
    parser.add_argument("--workers", type=int, default=None, help="并行线程数 (默认: 1; >20个文件时自动开启)")
    
    args = parser.parse_args()
    
    global client
    if args.key:
        client = Groq(api_key=args.key)
        
    if args.mode in ["all", "mp3_to_txt"]:
        if not client and not os.getenv("GROQ_API_KEY"):
            tqdm.write("\n" + "!"*60)
            tqdm.write("🛑 错误: 转录模式需要 Groq API Key。")
            tqdm.write("请设置环境变量 GROQ_API_KEY，或在命令中使用 --key 参数。")
            tqdm.write("如果你只想提取音频，无需 API Key，请使用默认模式或 --mode mp4_to_mp3。")
            tqdm.write("!"*60 + "\n")
            sys.exit(1)
            
    # 并行处理限制：如果开启多线程且是在转录模式下，提醒用户 Groq API 的并发限制
    if args.workers and args.workers > 1 and args.mode in ["all", "mp3_to_txt"]:
        tqdm.write("⚠️ 注意: 并行转录可能会更容易触发 Groq API 的频率限制 (RPM/TPM)。")

    process_directory(args.dir, args.mode, args.workers)
    tqdm.write("处理完成!")

if __name__ == "__main__":
    # 抑制 MacOS MallocStackLogging 噪音
    os.environ["MallocLogFile"] = "/dev/null"
    main()
