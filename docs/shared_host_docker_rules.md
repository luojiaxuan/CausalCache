# 共享 GPU 主机 Docker 规范(全会话统一,2026-07-23 定版)

适用于 luojiaxuan 名下所有会话(Claude / GPT)在 hyper00 / hyper01 / H100 等共享主机上的操作。
全局版已写入 ~/.claude/CLAUDE.md,本文件供各会话同步。

## 1. 命名:一律 sglang-omni-jaxan 前缀

- 容器名:`sglang-omni-jaxan-<MMDDHHMM>`,只带本地创建时间戳后缀,任何项目都一样;
  同一分钟批量创建多台时,时间戳顺序递增即可(如 07231000、07231001…)。
- **禁止**在容器名、镜像 tag、端口模式里出现项目名、benchmark 名、任务提示
  (causalcache / osworld / androidworld / assay / probe 之类全部不许)。
- 真实用途登记在两处:主机侧 `$HOME/jiaxuanluo-map.txt`(格式:`容器名 用途`)+ Git 文档。
- 端口用无特征高位段(如 28100+),不要 5000-5023 这种整齐排。

## 2. 镜像:统一匿名 tag

- policy 运行时:`jaxanluo/sglang-omni:dev`(原 causalcache-policy-ocr:v1)
- AndroidWorld 环境:`jaxanluo/sglang-omni:env`(原 causalcache-androidworld:11cea575)
- 三台主机已同步打 tag 并摘除全部泄底旧 tag;**启动命令一律用新名**。
- 新构建的镜像也照此规范起 tag(jaxanluo/sglang-omni:<短代号>),不带项目词。

## 3. 复用优先,不要每阶段起新容器

- 后续阶段优先 `docker exec` 进既有容器执行,或 `docker start` 复用停掉的同名容器;
- 长任务序列(采集→评分→评测)尽量在同一批容器内轮换命令,而不是一批一批地建新的。

## 4. 用完即清

- 容器用途结束**当场** `docker rm`;不积攒 exited 尸体、不留闲置 emulator
  (闲置 GPU 容器会被别人的 preflight 合法击杀,exited 容器会占名字并钉住镜像层)。
- 只清自己的(sglang-omni-jaxan-* 前缀);**别人的容器、运行中的任务绝不动**;
  共享机上不做全局 prune(docker system/image/volume prune 一律禁止)。

## 5. 跨机传文件的坑(实战教训)

- **不要**用带 NVIDIA CUDA banner 的镜像(jaxanluo/sglang-omni:dev 属于此类)`cat` 文件走管道
  ——banner 会打进 stdout 污染文件(torch checkpoint 会直接 UnpicklingError)。
- 正确做法:主机侧 `ssh A cat file | ssh B 'cat > file'` 直传,传完**双端 sha256 校验**;
  容器内需要 python 时用 ubuntu 镜像(无 banner)或落盘后再进容器。

## 6. GPU 使用惯例(既有规则重申)

- 发射前逐卡验占用(<1GB 才算空闲),GPU 快照存证进 run root;
- H100 是 80GB 卡:margin 训练/大 prompt 打分单卡单进程;H200(141GB)可 2-3 进程共卡;
- 别人的显存滞留进程(非容器)不清;自己的 0% 闲置容器按 5 秒采样规则清。
