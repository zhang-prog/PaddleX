# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import glob
import json
import os
import sys
import time
import uuid
from operator import itemgetter
from threading import Thread
import base64
import io
import pynvml
from tqdm import tqdm
from PIL import Image, ImageOps
import cv2

shutdown = False


class Predictor(object):
    def predict(self, task_info, batch_data):
        batch_data = self._preprocess(batch_data)
        task_info["start_time"] = get_curr_time()
        try:
            inference_time = self._predict(batch_data)
        except Exception as e:
            task_info["successful"] = False
            print(e)
            raise
        finally:
            task_info["end_time"] = get_curr_time()

        if isinstance(inference_time, float):  # doclayout_yolo
            task_info["start_time"] = 0
            task_info["end_time"] = inference_time * len(batch_data) / 1000.0  # ms -> s

        task_info["successful"] = True

    def _preprocess(self, input_files):
        raise NotImplementedError

    def _predict(self, batch_data):
        raise NotImplementedError

    def close(self):
        pass


class PaddleXPredictorTorch(Predictor):
    def __init__(self):
        super().__init__()

        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        model_path = "/PaddlePaddle/PP-DocLayoutV3_safetensors"
        self.model = (
            AutoModelForObjectDetection.from_pretrained(model_path).to("cuda").eval()
        )
        self.image_processor = AutoImageProcessor.from_pretrained(
            model_path, use_fast=True
        )

    def _preprocess(self, input_files):
        images = []
        for file in input_files:
            image = Image.open(file).convert("RGB")
            images.append(image)
        inputs = self.image_processor(images=images, return_tensors="pt").to("cuda")

        return inputs

    def _predict(self, batch_data):
        outputs = self.model(**batch_data, return_dict=True)


class PaddleXPredictor(Predictor):
    def __init__(self, device, batch_size, use_hpip, model_name):
        assert device in (
            "gpu",
            "dcu",
            "npu",
            "mlu",
            "gcu",
            "xpu",
            "iluvatar_gpu",
            "metax_gpu",
        ), f"{device} is not supported"
        assert model_name in (
            "PP-DocLayoutV2",
            "PP-DocLayoutV3",
        ), f"{model_name} is not supported, must in ['PP-DocLayoutV2', 'PP-DocLayoutV3']"
        from paddlex import create_model

        super().__init__()
        print("device: ", device)
        print("batch_size: ", batch_size)
        print("use_hpip: ", use_hpip)

        self.model = create_model(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            use_hpip=use_hpip,
        )

    def _preprocess(self, input_files):
        batch_data = self.model.batch_sampler(input_files)
        datas = next(batch_data).instances
        # preprocess
        for pre_op in self.model.pre_ops[:-1]:
            datas = pre_op(datas)

        # use `ToBatch` format batch inputs
        batch_inputs = self.model.pre_ops[-1](datas)

        return batch_inputs

    def _predict(self, batch_data):
        results = list(self.model.infer(batch_data))

    def close(self):
        self.model.close()


class DolphinPredictor(Predictor):
    def __init__(self, server_url, request_mode="fast_request"):
        super().__init__()

        self.request_mode = request_mode
        if request_mode == "fast_request":
            from mineru.backend.vlm.vlm_analyze import ModelSingleton

            self.model = ModelSingleton().get_model("http-client", None, server_url)
        else:
            from openai import OpenAI

            self.client = OpenAI(
                api_key="EMPTY",
                base_url=server_url,
            )
            self.model = self.client.models.list().data[0].id
            print("served model name", self.model)

    def _preprocess(self, input_files):
        if self.request_mode == "fast_request":
            images = []
            for image_path in input_files:
                img = Image.open(image_path).convert("RGB")
                images.append(img)

            batch_data = (
                images,
                ["Parse the reading order of this document."] * len(images),
                None,
                list(range(len(images))),
            )
            return batch_data
        else:
            all_messages = []
            for image_path in input_files:
                with Image.open(image_path) as img:
                    img = img.convert("RGB")
                    with io.BytesIO() as buf:
                        img.save(buf, format="PNG")
                        image_url = f"data:image/png;base64," + base64.b64encode(
                            buf.getvalue()
                        ).decode("ascii")
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_url}},
                            {
                                "type": "text",
                                "text": "Parse the reading order of this document.",
                            },
                        ],
                    }
                ]
                all_messages.append(messages)

            return all_messages

    def _predict(self, batch_data):
        if self.request_mode == "fast_request":
            layout_images, prompt, params, priority = batch_data
            outputs = self.model.client.batch_predict(
                layout_images, prompt, params, priority
            )
        else:
            for data in batch_data:
                chat_completion_from_url = self.client.chat.completions.create(
                    messages=data,
                    model=self.model,
                    temperature=0.0,
                )


class MinerUPredictor(Predictor):
    def __init__(self, server_url):
        from mineru.backend.vlm.vlm_analyze import ModelSingleton

        super().__init__()
        self.model = ModelSingleton().get_model("http-client", None, server_url)

    def _preprocess(self, input_files, priority=None):
        images = []
        for image_path in input_files:
            image = Image.open(image_path).convert("RGB")
            image = ImageOps.exif_transpose(image) or image
            images.append(image)

        if priority is None and self.model.incremental_priority:
            priority = list(range(len(images)))

        layout_images = self.model.helper.batch_prepare_for_layout(
            self.model.executor, images
        )
        prompt = self.model.prompts.get("[layout]") or self.prompts["[default]"]
        params = self.model.sampling_params.get("[layout]") or self.sampling_params.get(
            "[default]"
        )

        batch_data = (layout_images, prompt, params, priority)
        return batch_data

    def _predict(self, batch_data):
        # results = self.predictor.batch_layout_detect(images=batch_data)
        layout_images, prompt, params, priority = batch_data
        outputs = self.model.client.batch_predict(
            layout_images, prompt, params, priority
        )


class DocLayoutYOLOPredictor(Predictor):
    def __init__(
        self,
        model_path=None,
        imgsz: int = 1024,
        conf_threshold: float = 0.2,
        iou_threshold: float = 0.45,
    ):
        """
        初始化DocLayout-YOLO模型

        Args:
            model_path: 模型权重路径，None则自动下载
            imgsz: 输入图像尺寸
            conf_threshold: 置信度阈值
            iou_threshold: NMS IoU阈值
        """
        from doclayout_yolo import YOLOv10

        super().__init__()

        self.imgsz = imgsz
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        if model_path is None:
            model_path = self._auto_download_model()

        print(f"正在加载DocLayout-YOLO模型: {model_path}")
        self.model = YOLOv10(model_path)

    def _auto_download_model(self) -> str:
        """
        自动下载模型（参考MinerU的模型下载逻辑）
        """
        try:
            from modelscope import snapshot_download

            cache_dir = os.environ.get("MODELSCOPE_CACHE", "./models")
            model_dir = snapshot_download(
                "opendatalab/DocLayout-YOLO", cache_dir=cache_dir
            )
            # 查找最佳模型文件
            for root, dirs, files in os.walk(model_dir):
                for file in files:
                    if file.endswith(".pt") or file.endswith(".pth"):
                        return os.path.join(root, file)
            raise FileNotFoundError("未找到模型权重文件")
        except Exception as e:
            print(f"自动下载失败: {e}")
            raise RuntimeError("请手动指定模型路径或安装modelscope")

    def _preprocess(self, input_files):
        images = []
        for file in input_files:
            image = cv2.imread(file)
            images.append(image)

        return batch_data

    def _predict(self, batch_data):
        results = self.model.predict(
            batch_data,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device="cuda:1",
            # verbose=True,
            # visualize=False,
            # save_txt=True,
            # save_dir="./"
        )
        return results[0].speed["inference"]


def monitor_device(gpu_ids, gpu_metrics_list):
    try:
        pynvml.nvmlInit()
        handles = [pynvml.nvmlDeviceGetHandleByIndex(gpu_id) for gpu_id in gpu_ids]

        time.sleep(5)
        while not shutdown:
            try:
                gpu_util = 0
                mem_bytes = 0

                for handle in handles:
                    gpu_util += pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                    mem_bytes += pynvml.nvmlDeviceGetMemoryInfo(handle).used

                gpu_metrics_list.append(
                    {
                        "utilization": gpu_util,
                        "memory": mem_bytes,
                    }
                )
            except Exception as e:
                print(f"Error monitoring GPUs: {e}")

            time.sleep(0.5)

    except Exception as e:
        print(f"Error initializing the GPU monitor: {e}")
    finally:
        try:
            pynvml.nvmlShutdown()
        except:
            pass


def get_curr_time():
    return time.perf_counter()


def new_task_info():
    task_info = {}
    task_info["id"] = uuid.uuid4().hex
    return task_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dirs", type=str, nargs="+", metavar="INPUT_DIR")
    parser.add_argument(
        "-t",
        "--predictor_type",
        type=str,
        default="paddlex",
        choices=["paddlex", "paddlex_torch", "mineru", "dolphin", "doclayoutyolo"],
    )
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("-o", "--output_path", type=str, default="benchmark.json")
    parser.add_argument("--device", type=str, default="gpu")  # paddlex device
    parser.add_argument(
        "--use_hpip",
        action="store_true",
        help="Enable the use of HPIP. (Default: disabled)",
    )
    parser.add_argument(
        "--paddlex_model_name", type=str, default="PP-DocLayoutV3"
    )  # paddlex model name
    parser.add_argument(
        "--mineru_server_url", type=str, default="http://127.0.0.1:30000"
    )
    parser.add_argument(
        "--dolphin_server_url", type=str, default="http://127.0.0.1:8000/v1"
    )
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0])
    args = parser.parse_args()

    task_info_list = []

    all_input_paths = []
    for input_dir in args.input_dirs:
        all_input_paths += glob.glob(os.path.join(input_dir, "*"))
    all_input_paths.sort()
    if len(all_input_paths) == 0:
        print("No valid data")
        sys.exit(1)

    all_input_paths = all_input_paths

    if args.predictor_type == "paddlex":
        predictor = PaddleXPredictor(
            args.device, args.batch_size, args.use_hpip, args.paddlex_model_name
        )
    elif args.predictor_type == "paddlex_torch":
        predictor = PaddleXPredictorTorch(args.batch_size)
    elif args.predictor_type == "doclayoutyolo":
        predictor = DocLayoutYOLOPredictor()
    elif args.predictor_type == "mineru":
        predictor = MinerUPredictor(args.mineru_server_url)
    elif args.predictor_type == "dolphin":
        predictor = DolphinPredictor(args.dolphin_server_url)
    else:
        print(f"Unsupported predictor type: {args.predictor_type}")
        sys.exit(2)

    if args.batch_size < 1:
        print("Invalid batch size")
        sys.exit(2)

    gpu_metrics_list = []
    thread_device_monitor = Thread(
        target=monitor_device,
        args=(args.gpu_ids, gpu_metrics_list),
    )
    thread_device_monitor.start()

    try:
        batch_data = []
        for i, input_path in tqdm(
            enumerate(all_input_paths), total=len(all_input_paths)
        ):
            batch_data.append(input_path)
            if len(batch_data) == args.batch_size or i == len(all_input_paths) - 1:
                task_info = new_task_info()
                predictor.predict(task_info, batch_data)
                task_info_list.append(task_info)
                batch_data.clear()
    finally:
        shutdown = True
        thread_device_monitor.join()
        predictor.close()

    total_files = len(all_input_paths)
    duration_list_batch = [
        info["end_time"] - info["start_time"] for info in task_info_list
    ]
    avg_latency_batch = (
        sum(duration_list_batch) / len(duration_list_batch)
        if duration_list_batch
        else 0
    )

    print(f"Total elapsed time: {sum(duration_list_batch) * 1000.0:.4f} ms")
    print(f"Batch elapsed time: {avg_latency_batch * 1000.0:.4f} ms")
    print(
        f"Time per image: {sum(duration_list_batch) / len(all_input_paths) * 1000.0:.4f} ms"
    )

    successful_files = sum(map(lambda x: x["successful"], task_info_list))
    if gpu_metrics_list:
        gpu_util_list = list(map(itemgetter("utilization"), gpu_metrics_list))
        print(
            f"GPU utilization (%): {max(gpu_util_list):.1f}, {min(gpu_util_list):.1f}, {sum(gpu_util_list) / len(gpu_util_list):.1f}"
        )
        gpu_mem_list = list(map(itemgetter("memory"), gpu_metrics_list))
        print(
            f"GPU memory usage (MB): {max(gpu_mem_list) / 1024**2:.1f}, {min(gpu_mem_list) / 1024**2:.1f}, {sum(gpu_mem_list) / len(gpu_mem_list) / 1024**2:.1f}"
        )

    dic = {
        "input_dirs": args.input_dirs,
        "predictor_type": args.predictor_type,
        "batch_size": args.batch_size,
        "total_files": total_files,
        "time_per_image": sum(duration_list_batch) / len(all_input_paths),
        "avg_latency_batch": avg_latency_batch,
        "duration_list": duration_list_batch,
        "successful_files": successful_files,
        "gpu_metrics_list": gpu_metrics_list,
    }
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(
            dic,
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Config and results saved to {args.output_path}")
