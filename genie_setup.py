"""
GENIE 模型设置脚本
用于帮助用户设置和转换自定义GENIE模型
"""
import os
import sys
from pathlib import Path

try:
    import genie_tts as genie
    GENIE_AVAILABLE = True
except ImportError:
    GENIE_AVAILABLE = False
    genie = None


def setup_genie_data_dir(data_dir: str):
    """
    设置GENIE数据目录
    """
    if not GENIE_AVAILABLE:
        print("错误: 未安装 genie-tts，请先运行 'pip install genie-tts'")
        return False
    
    os.environ["GENIE_DATA_DIR"] = data_dir
    print(f"GENIE 数据目录已设置为: {data_dir}")
    return True


def load_predefined_character(character_name: str):
    """
    加载预定义角色
    """
    if not GENIE_AVAILABLE:
        print("错误: 未安装 genie-tts，请先运行 'pip install genie-tts'")
        return False
    
    try:
        genie.load_predefined_character(character_name)
        print(f"成功加载预定义角色: {character_name}")
        return True
    except Exception as e:
        print(f"加载预定义角色 {character_name} 失败: {e}")
        return False


def load_custom_character(character_name: str, model_dir: str, language: str = "jp"):
    """
    加载自定义角色模型
    """
    if not GENIE_AVAILABLE:
        print("错误: 未安装 genie-tts，请先运行 'pip install genie-tts'")
        return False
    
    try:
        genie.load_character(
            character_name=character_name,
            onnx_model_dir=model_dir,
            language=language
        )
        print(f"成功加载自定义角色: {character_name} 从 {model_dir}")
        return True
    except Exception as e:
        print(f"加载自定义角色 {character_name} 失败: {e}")
        return False


def set_reference_audio(character_name: str, audio_path: str, audio_text: str):
    """
    设置参考音频
    """
    if not GENIE_AVAILABLE:
        print("错误: 未安装 genie-tts，请先运行 'pip install genie-tts'")
        return False
    
    try:
        genie.set_reference_audio(
            character_name=character_name,
            audio_path=audio_path,
            audio_text=audio_text
        )
        print(f"成功设置 {character_name} 的参考音频: {audio_path}")
        return True
    except Exception as e:
        print(f"设置参考音频失败: {e}")
        return False


def test_tts(character_name: str, text: str, output_path: str = None):
    """
    测试TTS功能
    """
    if not GENIE_AVAILABLE:
        print("错误: 未安装 genie-tts，请先运行 'pip install genie-tts'")
        return False
    
    try:
        if output_path:
            genie.tts(
                character_name=character_name,
                text=text,
                save_path=output_path
            )
            print(f"TTS 测试完成，音频已保存到: {output_path}")
        else:
            genie.tts(
                character_name=character_name,
                text=text,
                play=True
            )
            print("TTS 测试完成，音频已播放")
        
        return True
    except Exception as e:
        print(f"TTS 测试失败: {e}")
        return False


def convert_model(torch_pth_path: str, torch_ckpt_path: str, output_dir: str):
    """
    转换原始GPT-SoVITS模型为GENIE格式
    """
    if not GENIE_AVAILABLE:
        print("错误: 未安装 genie-tts，请先运行 'pip install genie-tts'")
        return False
    
    try:
        genie.convert_to_onnx(
            torch_pth_path=torch_pth_path,
            torch_ckpt_path=torch_ckpt_path,
            output_dir=output_dir
        )
        print(f"模型转换完成，输出到: {output_dir}")
        return True
    except Exception as e:
        print(f"模型转换失败: {e}")
        return False


def main():
    """
    主函数，提供交互式界面
    """
    if not GENIE_AVAILABLE:
        print("错误: 未安装 genie-tts，请先运行 'pip install genie-tts'")
        return

    print("GENIE 模型设置助手")
    print("=" * 40)
    print("1. 设置数据目录")
    print("2. 加载预定义角色")
    print("3. 加载自定义角色")
    print("4. 设置参考音频")
    print("5. 测试TTS")
    print("6. 转换模型")
    print("0. 退出")
    print("=" * 40)
    
    while True:
        try:
            choice = input("\n请选择操作 (0-6): ").strip()
            
            if choice == "0":
                print("退出程序")
                break
            elif choice == "1":
                data_dir = input("请输入GENIE数据目录路径: ").strip()
                setup_genie_data_dir(data_dir)
            elif choice == "2":
                char_name = input("请输入预定义角色名 (如: mika, 37, feibi): ").strip()
                load_predefined_character(char_name)
            elif choice == "3":
                char_name = input("请输入自定义角色名: ").strip()
                model_dir = input("请输入模型目录路径: ").strip()
                language = input("请输入语言代码 (jp/zh/en，默认jp): ").strip() or "jp"
                load_custom_character(char_name, model_dir, language)
            elif choice == "4":
                char_name = input("请输入角色名: ").strip()
                audio_path = input("请输入参考音频路径: ").strip()
                audio_text = input("请输入参考音频对应的文本: ").strip()
                set_reference_audio(char_name, audio_path, audio_text)
            elif choice == "5":
                char_name = input("请输入角色名: ").strip()
                text = input("请输入要合成的文本: ").strip()
                output_path = input("输出路径 (留空则直接播放): ").strip() or None
                test_tts(char_name, text, output_path)
            elif choice == "6":
                pth_path = input("请输入.pth模型文件路径: ").strip()
                ckpt_path = input("请输入.ckpt检查点文件路径: ").strip()
                output_dir = input("请输入输出目录: ").strip()
                convert_model(pth_path, ckpt_path, output_dir)
            else:
                print("无效选择，请重新输入")
        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
            break
        except Exception as e:
            print(f"操作失败: {e}")


if __name__ == "__main__":
    main()