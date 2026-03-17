#!/bin/bash
set -e
export http_proxy
export https_proxy

echo "commit_id: $commit_id"
apt install git-lfs
git clone --depth 1 https://github.com/kairos-agi/kairos-sensenova.git /root/code/kairos-sensenova
cd /root/code/kairos-sensenova
git fetch --depth 1 origin "$commit_id"
git checkout "$commit_id"
git lfs pull
ls -alh
# 测试运行的命令
echo "Run cmd: {$case_cmd}"
$case_cmd
ls output -alh

# 产物上传
# download ads-cli
if [ -f /KAIROS_vepfs-2/KAIROS_vepfs/fuzuoyi/adscli/ads-cli ]; then
    echo "Copying ads-cli ..."
    cp /KAIROS_vepfs-2/KAIROS_vepfs/fuzuoyi/adscli/ads-cli /usr/bin/
    chmod +x /usr/bin/ads-cli
elif ! command -v ads-cli &> /dev/null; then
    echo "Downloading ads-cli..."
    wget -q https://quark.aoss.cn-sh-01.sensecoreapi-oss.cn/ads-cli/release/v1.10.0/ads-cli
    chmod +x ads-cli
    mv ads-cli /usr/bin/
fi

remote_endpoint="white-bucket.aoss.cn-sh-01b.sensecoreapi-oss.cn"
ads-cli -q cp  output/*/* s3://$ak:$sk@$remote_endpoint/$remote_dir
echo "Upload output to $remote_dir..."
