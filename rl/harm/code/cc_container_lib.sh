# note (luojiaxuan): 共享主机容器纪律的公共函数(source 进各发射脚本):
#   alloc_name  取 docker ps -a 与 map 两边都空缺的最小编号;
#   reg         创建后立即登记 map;
#   rm_own      只删 cc.owner 标签等于本次 OWN 的容器,其它跳过并打印;
#   pick_gpu    选 <2GB 占用、且未被 map 中仍存在的容器登记的卡(可传入要排除的卡号)。
MAP=$HOME/jiaxuanluo-map.txt
OWN="cc-$(hostname)-$(date -u +%s)-$$"
alloc_name() { local used n; used=$( { docker ps -a --format '{{.Names}}'; cut -f1 "$MAP"; } | grep -o '^sglang-omni-jaxan-[0-9]\+$' | sed 's/.*-//' | sort -un); for n in $(seq 1 99); do grep -qx "$n" <<<"$used" || { echo "sglang-omni-jaxan-$n"; return; }; done; }
rm_own() { [ -n "${1:-}" ] || return 0; [ "$(docker inspect -f '{{index .Config.Labels "cc.owner"}}' "$1" 2>/dev/null)" = "$OWN" ] || { echo "SKIP rm $1 (not mine)"; return 0; }; docker rm -f "$1" >/dev/null 2>&1; sed -i "/^$1\t/d" "$MAP"; }
reg() { printf "%s\t%s\t%s\t%s\t%s\n" "$1" "gpus=$2" "host=$(hostname)" "created=$(date -u +%FT%TZ)" "desc=$3" >> "$MAP"; }
pick_gpu() { local claimed ex; ex="${1:-}"; claimed=$(join <(docker ps --format '{{.Names}}' | sort) <(sort "$MAP") -t $'\t' 2>/dev/null | cut -f2 | sed 's/gpus=//; s/idx//g' | tr ',' '\n' | grep -x '[0-9]\+' | sort -u); nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<2000{print $1}' | while read g; do grep -qx "$g" <<<"$claimed" && continue; grep -qw "$g" <<<"$ex" && continue; echo "$g"; break; done; }
# note (luojiaxuan): 并行 session 同秒分配同一个名字/端口会撞:端口从 41300 起扫空闲;docker run 失败则换名重试 3 次。
free_port() { local p; for p in $(seq 41300 41399); do ss -ltn 2>/dev/null | grep -q ":$p " || { echo "$p"; return; }; done; }
run_labeled() { local i; for i in 1 2 3; do C=$(alloc_name); if docker run -d --init --label cc.owner=$OWN --name "$C" "$@" >/dev/null 2>&1; then return 0; fi; sleep $((RANDOM % 5 + 1)); done; C=""; return 1; }
