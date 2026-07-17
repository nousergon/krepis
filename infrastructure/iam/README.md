# krepis-runner-executor-role setup

Companion to the krepis-runner-dispatcher Lambda (nousergon-data). Two-step
provisioning, both required — step 2 was a real bug found live 2026-07-17
during the metron dispatch (boxes launched fine but never registered with
SSM at all, sat "waiting_ssm" until the bootstrap deadline reaped them).

```sh
aws iam create-role --role-name krepis-runner-executor-role \
  --assume-role-policy-document file://infrastructure/iam/krepis-runner-executor-role-trust.json
aws iam put-role-policy --role-name krepis-runner-executor-role \
  --policy-name krepis-runner-executor-policy \
  --policy-document file://infrastructure/iam/krepis-runner-executor-role-policy.json
aws iam create-instance-profile --instance-profile-name krepis-runner-executor-profile
aws iam add-role-to-instance-profile --instance-profile-name krepis-runner-executor-profile \
  --role-name krepis-runner-executor-role

# REQUIRED — without this the SSM agent on the box can never call
# ssm:UpdateInstanceInformation/ssmmessages:*/ec2messages:* to register
# itself with Systems Manager at all. See metron#276 for the full incident.
aws iam attach-role-policy --role-name krepis-runner-executor-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
```
