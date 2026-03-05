# 通用的server-client

## /Simple_Remote_Infer/deploy/qwenpi_policy.py

- QwenPiServer: 一个用于示范的模型，拥有init和infer方法

将加载好的模型用`WebsocketPolicyServer`包裹，并指定端口即可

```python
model_server = WebsocketPolicyServer(model, port=8002)
model_server.serve_forever() # 开启监听
```

## ./websocket_client_policy.py

在`__main__()`中展示了如何创造一个假模型向真模型发送环境信息，只需要用`WebsocketClientPolicy`代替原有的模型即可

## ./deploy/remote_inference_lerobot_norobot.py

新增了一个“无真机”客户端测试脚本，参考了 `flexiv_rdk/src/flexiv/inference/remote_inference_lerobot_norobot.py`，并适配了本仓的 `Simple_Remote_Infer` websocket 框架。

示例：

```bash
python -m wan_va.utils.Simple_Remote_Infer.deploy.remote_inference_lerobot_norobot \
  --policy_host 127.0.0.1 \
  --policy_port 1106 \
  --prompt "pick up the object"
```

## ./deploy/remote_inference_lerobot_realrobot.py

新增了一个真机推理脚本（Flexiv + 3路 RealSense），参考 `flexiv_rdk/src/flexiv/inference/remote_inference_lerobot.py`，但通信协议适配为当前 LingBot-VA server 的：
- `{"reset": true, "prompt": ...}`
- `{"obs": ...}` 推理 action chunk
- `{"obs": key_frames, "compute_kv_cache": true, "state": action_chunk}`

示例：

```bash
python -m wan_va.utils.Simple_Remote_Infer.deploy.remote_inference_lerobot_realrobot \
  --left_robot_sn <LEFT_SN> \
  --right_robot_sn <RIGHT_SN> \
  --policy_camera1_sn <CAM1_SN> \
  --policy_camera2_sn <CAM2_SN> \
  --policy_camera3_sn <CAM3_SN> \
  --policy_host 127.0.0.1 \
  --policy_port 1106 \
  --prompt "pick up the object"
```
