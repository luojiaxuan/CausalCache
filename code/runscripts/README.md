# runscripts:主机侧运行脚本存档(可复用/可改)

按用户 2026-07-30 指示:每次跑实验的主机脚本入 Git,不再只存散在 /tmp。
命名 = 当时的实际文件名;路径硬编码对应各主机容器布局(h00=/data,
h01=/bigdata,详见 docs/run-mobileworld-review-ablations.md)。

- OSWorld 闭环链:osworld_serve_loop_h00.sh(serve_osworld_official_policy,
  adapter/selector 开关)+ osworld_worker_loop_h00.sh
  (run_osworld_benchmark_worker,--max-steps 默认 15,30 步诊断即改此参)。
- MobileWorld 32B 两臂:launch_32b_h00/h01.sh;8B selector 闭环:
  launch_frozensel_ft.sh / launch_selv5.sh;翻臂:flip_h00/h01.sh。
- 训练/评分:train_v5a.sh、train_32b_hgkv.sh、devscore_*.sh、
  train_frozen_selector.sh、frozen_singletons_h00.sh、fsets_hyper00.sh。
- 收割:harvest_arm.sh(双机 task→score 去重)。

教训三条(此目录脚本已体现):宿主 /tmp ≠ 容器 /tmp;docker exec 内
nohup 不保活(用 docker exec -d);监督器发射后不可动,server 无状态可随时迁卡。
