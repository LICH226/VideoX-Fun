import wandb
import numpy as np
from PIL import Image
import os

# 设置为离线模式，这样测试快，不用联网上传
os.environ["WANDB_MODE"] = "offline"

def test_wandb_image_logging():
    print("🚀 开始测试 wandb.Image 功能...")

    # 1. 初始化一个临时的 run
    try:
        run = wandb.init(project="test-wandb-image", name="test-run")
        print("✅ wandb 初始化成功。")
    except Exception as e:
        print(f"❌ wandb 初始化失败: {e}")
        return

    # 2. 创建一个假的图片 (一个红色的正方形)
    # 创建一个 [100, 100, 3] 的 numpy 数组，全填满红色 [255, 0, 0]
    data = np.zeros((100, 100, 3), dtype=np.uint8)
    data[:] = [255, 0, 0] 
    pil_image = Image.fromarray(data, 'RGB')
    print("✅ 测试用 PIL Image 创建成功。")

    # 3. 测试 wandb.Image 和 log
    try:
        # 关键：创建 wandb Image 对象
        w_image = wandb.Image(pil_image, caption="This is a test image")
        print("✅ wandb.Image 对象创建成功。")

        # 关键：记录 log
        wandb.log({"test_validation_sample": w_image}, step=1)
        print("✅ wandb.log 执行成功。")
        
    except NameError as e:
        print(f"❌ 捕获到 NameError: {e}")
        print("这说明 wandb 没有被正确导入。")
    except UnboundLocalError as e:
        print(f"❌ 捕获到 UnboundLocalError: {e}")
        print("这说明函数内部有局部变量遮蔽了全局的 wandb。")
    except Exception as e:
        print(f"❌ 捕获到其他错误: {e}")

    # 4. 结束 run
    wandb.finish()
    print("🏁 测试结束。如果上面全是✅，说明 wandb.Image 功能本身没问题。")
    print(f"离线日志保存在: {run.dir}")

if __name__ == "__main__":
    test_wandb_image_logging()