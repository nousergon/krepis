set -eo pipefail
export HOME=/home/ec2-user XDG_CACHE_HOME=/tmp AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1
export S3_STAGING=s3://alpha-engine-research/staging/x

if ! systemctl is-enabled ec2-spot-watchdog 2>/dev/null; then
  cat > /etc/systemd/system/ec2-spot-watchdog.service <<'UNIT'
[Unit]
Description=EC2 Spot Watchdog — self-terminate on SSM agent stoppage
After=amazon-ssm-agent.service
Requires=amazon-ssm-agent.service

[Service]
Type=simple
ExecStart=/usr/local/bin/ec2-spot-watchdog.sh
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
UNIT
  cat > /usr/local/bin/ec2-spot-watchdog.sh <<'WDSH'
#!/usr/bin/env bash
set -euo pipefail
while true; do
  if ! systemctl is-active amazon-ssm-agent >/dev/null 2>&1; then
    sleep 60
    if ! systemctl is-active amazon-ssm-agent >/dev/null 2>&1; then
      shutdown -h now
    fi
  fi
  sleep 60
done
WDSH
  chmod +x /usr/local/bin/ec2-spot-watchdog.sh
  timeout 60 systemctl enable --now ec2-spot-watchdog || {
    echo "ERROR: enabling ec2-spot-watchdog did not return within 60s — the unit is misdeclared (an endless ExecStart under Type=oneshot blocks systemctl start forever)" >&2
    exit 1
  }
fi

systemd-run --on-active=3600 --unit=ec2-spot-hard-timeout \
    --description='spot hard runtime cap (3600s)' /sbin/shutdown -h now || {
  echo "ERROR: could not arm the 3600s hard-timeout timer — refusing to start an uncapped spot workload" >&2
  exit 1
}

dnf install -y -q python3.12 python3.12-pip python3.12-devel git gcc 2>/dev/null || \
    dnf install -y -q python3 python3-pip python3-devel git gcc
command -v python3.12 >/dev/null || { echo "ERROR: python3.12 not found after dnf install" >&2; exit 1; }
echo "Using: $(python3.12 --version)"

if [ ! -d /home/ec2-user/data/.git ]; then
  rm -rf /home/ec2-user/data
  git clone --depth 1 --branch main https://github.com/nousergon/nousergon-data.git /home/ec2-user/data
fi
if [ ! -d /home/ec2-user/executor/.git ]; then
  rm -rf /home/ec2-user/executor
  git clone --depth 1 --branch main https://github.com/nousergon/crucible-executor.git /home/ec2-user/executor
fi

mkdir -p /home/ec2-user/data
aws s3 cp "${S3_STAGING}/config.yaml" /home/ec2-user/data/config.yaml --region us-east-1 --quiet
if [ "${STAGED}" = "1" ]; then
  mkdir -p /opt/p
  aws s3 cp "${S3_STAGING}/p.json" /opt/p/p.json --region us-east-1 --quiet
  chown -R ec2-user:ec2-user /opt/p
else
  echo "SKIPPED: p.json not staged (condition ${STAGED} was not 1) — /opt/p/p.json will not exist"
fi
