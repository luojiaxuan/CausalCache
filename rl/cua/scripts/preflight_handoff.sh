#!/usr/bin/env bash
# note (luojiaxuan): 接手方环境前置自检。在开始 §2 环境自举之前跑一遍,
# 把会让人白忙半天的硬门槛一次验完。只读,不改机器任何状态。
echo "=== CausalCache RL 交接前置自检 ==="
fail=0; warn=0
ck() { if [ "$2" = ok ]; then echo "  [OK]   $1"; else echo "  [$2] $1"; [ "$2" = FAIL ] && fail=$((fail+1)) || warn=$((warn+1)); fi; }

echo "--- 1. GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/  /'
  N=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
  MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  [ "$N" -ge 3 ] && ck "GPU 数 $N ≥ 3(1 rollout + 2 train 最小形态)" ok \
                 || ck "GPU 数 $N < 3:单卡跑不动本管线(需 ≥3,推荐 4+)" FAIL
  if [ "$MEM" -lt 90000 ]; then
    ck "单卡 ${MEM}MiB < 90GB:TP2 峰值约 90GB/卡 → **必须 TP_SIZE=4 起步**(见 handoff §5.1)" WARN
  else
    ck "单卡显存 ${MEM}MiB(TP2 可行)" ok
  fi
  ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
  case "$ARCH" in
    8.0|8.6) ck "计算能力 $ARCH(Ampere:bf16 可用,FP8 不可用——我方本就用 bf16,无影响)" ok ;;
    9.0|10.*) ck "计算能力 $ARCH(Hopper+)" ok ;;
    *) ck "计算能力 $ARCH(未在我方验证)" WARN ;;
  esac
else
  ck "nvidia-smi 不存在" FAIL
fi

echo "--- 2. KVM(MobileWorld emulator 的硬门槛)"
if [ -e /dev/kvm ]; then
  if [ -r /dev/kvm ] && [ -w /dev/kvm ]; then ck "/dev/kvm 可读写" ok
  else ck "/dev/kvm 存在但当前用户无读写权(需加入 kvm 组或 root)" FAIL; fi
else
  ck "/dev/kvm 不存在:若本机是虚拟机需开嵌套虚拟化,否则 MobileWorld emulator 无法启动(整条 env 线不可用)" FAIL
fi
grep -qE '^flags.*(vmx|svm)' /proc/cpuinfo 2>/dev/null && ck "CPU 虚拟化扩展可见" ok || ck "CPU 无 vmx/svm 标志" WARN

echo "--- 3. Docker"
if command -v docker >/dev/null 2>&1 && docker ps >/dev/null 2>&1; then
  ck "docker 可用(当前用户有权限)" ok
  docker info --format '   存储根: {{.DockerRootDir}}' 2>/dev/null
  df -h "$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)" 2>/dev/null | tail -1 | awk '{print "   可用空间: "$4" (建议 ≥500G:镜像+emulator 卷)"}'
else
  ck "docker 不可用或当前用户无权限" FAIL
fi

echo "--- 4. CPU / 内存(emulator 按 2.5 核/台预算)"
C=$(nproc); M=$(free -g 2>/dev/null | awk 'NR==2{print $2}')
[ "$C" -ge 32 ] && ck "CPU $C 核(可支撑 ~$((C/3)) 台 emulator)" ok || ck "CPU 仅 $C 核:env 并发会很低(rollout-bound,是第一瓶颈)" WARN
[ "${M:-0}" -ge 200 ] && ck "内存 ${M}G" ok || ck "内存 ${M}G:训练容器建议 ≥400g(optimizer offload 吃主存)" WARN

echo "--- 5. 网络可达(全部公开源)"
for u in https://github.com/cua-lite/cua-lite https://huggingface.co https://ghcr.io https://registry-1.docker.io; do
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 10 "$u" 2>/dev/null)
  [ "$code" != "000" ] && ck "$u ($code)" ok || ck "$u 不可达(需代理/镜像源)" FAIL
done

echo "--- 6. 工具链"
for t in git uv python3; do
  command -v $t >/dev/null 2>&1 && ck "$t $( $t --version 2>&1 | head -1 )" ok || ck "$t 缺失" FAIL
done

echo
echo "=== 结论: FAIL=$fail WARN=$warn ==="
[ $fail -gt 0 ] && echo "有 FAIL 项 → 先解决再开始 handoff §2;/dev/kvm 与 GPU 数是不可绕过的两条。"
[ $fail -eq 0 ] && echo "前置通过 → 按 rl/docs/handoff_h20.md §2 执行环境自举。"
exit 0
