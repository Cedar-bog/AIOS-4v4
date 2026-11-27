# benchmark

AIOS 调度对比实验的测量环境。用假 LLM 后端替换最底层的一次模型调用，调度器、队列、线程池、syscall 通知都走真实代码，用来观察不同调度策略下的排队、执行、通知和响应时间。

## 目录

```
benchmark/
  query.py          BenchLLMQuery，给请求挂实验字段
  scenario.py       场景生成，算好每个请求的到达时间和服务时间
  mock_backend.py   假 LLM 后端，按场景睡眠
  tracer.py         事件写入 JSONL
  workload.py       按场景发请求，支持多步任务
  runner.py         实验入口
  metrics.py        指标汇总
  selftest.py       环境自检
  configs/          场景配置
```

## 快速开始

先跑环境自检：

```bash
python -m benchmark.selftest
```

跑一个冒烟实验：

```bash
python -m benchmark.runner --config benchmark/configs/smoke.yaml --out results/smoke
```

跑长短混合负载，重复三次：

```bash
python -m benchmark.runner --config benchmark/configs/mixed.yaml --out results/mixed --repeats 3
```

覆盖批次间隔：

```bash
python -m benchmark.runner --config benchmark/configs/mixed.yaml --out results/mixed_b01 --batch-interval 0.1
```

## 输出

```
results/mixed/
  scenario.jsonl        场景，一次生成，所有重复共用
  experiment.json       配置和每次运行的概况
  run_00/
    events.jsonl        原始事件
    run_meta.json       本次运行的概况
  requests.csv          每个请求的指标
  summary.csv           按分组的统计表
  summary.json          完整统计结果
```

## 时间点

每个请求会记录七个时间点，单位是 `time.monotonic()`。

| 事件 | 含义 |
| --- | --- |
| arrive | 生成器准备发起请求 |
| enqueue | 请求进入 LLM 队列 |
| batch | 调度器把请求取出，准备成批执行 |
| exec_start | 后端工作线程真正开始执行 |
| backend_done | 假后端结束睡眠 |
| notify | 结果写回，调用方被唤醒 |
| return | 调用方拿到结果 |

由此得到的指标：

| 指标 | 计算方式 |
| --- | --- |
| enqueue_gap_ms | enqueue - arrive |
| batch_wait_ms | batch - enqueue，等待被调度器取走 |
| sched_wait_ms | exec_start - batch，从取走到真正执行 |
| service_ms | backend_done - exec_start，后端实际耗时 |
| notify_lag_ms | notify - backend_done，结果写回延迟 |
| response_ms | return - arrive，调用方看到的总时长 |
| slowdown | response_ms / service_ms |

`batch_wait_ms` 和 `notify_lag_ms` 是这次要优化的两个主要对象。原版里前者来自批次边界，后者来自同组统一返回。

## 配置字段

```yaml
name: mixed            # 实验名
seed: 42               # 随机种子，决定到达序列和服务时间
num_tasks: 200         # 任务数
batch_interval: 0.1    # 调度器批次间隔，秒

steps:                 # 每个任务的步数
  min: 1
  max: 1
  gap_ms: 200          # 多步任务中上一步返回后等待多久再发下一步

arrival:
  pattern: poisson     # uniform、poisson、burst、all_at_once
  rate_per_sec: 40     # uniform 和 poisson 使用
  jitter: 0.5          # uniform 的间隔抖动比例

burst:                 # pattern 为 burst 时使用
  size: 50
  gap_ms: 300

prompt:                # 输入 token 范围
  tokens_min: 50
  tokens_max: 1500

output:                # 输出 token 分布，对数正态
  median_tokens: 80
  sigma: 1.0
  min_tokens: 1

max_tokens: 512        # 输出上限

model:                 # 服务时间模型
  base_ms: 20
  prefill_ms_per_token: 0.02
  decode_ms_per_token: 5.0
  jitter_sigma: 0.15

faults:
  error_rate: 0.0      # 返回错误的比例
  timeout_rate: 0.0    # 返回超时的比例
  slow_rate: 0.0       # 额外延时的比例
  slow_extra_ms: 3000
```

服务时间在场景生成阶段算好并写盘：

```
service_ms = base_ms
           + prefill_ms_per_token * prompt_tokens
           + decode_ms_per_token * output_tokens
           + 对数正态抖动
```

不同策略跑同一份场景时，每个请求的耗时完全一致。

## 多步任务的到达时间

单步负载的到达序列可以完全预生成，不同策略复用同一份序列。

多步负载做不到这一点。第 k+1 步的到达时间取决于第 k 步什么时候返回，而返回时间本身受调度策略影响。场景文件只保证每个任务的启动时间和每个请求的服务时间一致，步骤之间的间隔从配置读取。报告里比较多步负载时需要注意这个差别。

## 新增一种调度策略

1. 在 `aios/scheduler/` 下实现新的调度器，接口参考 `FIFOScheduler`。
2. 在 `benchmark/runner.py` 的 `build_scheduler` 里加一个分支，或者加一个 `--policy` 参数。
3. 其余部分不用改，事件、负载和指标都是通用的。

## 注意事项

- 实验运行时会把日志级别降到 WARNING，并把调度器日志设为 silent，避免同步 I/O 污染时间戳。
- 原版没有并发上限，`peak_concurrency` 会等于同时在跑的请求数。加并发上限之后，这个指标用来看上限是否生效。
- `results/` 目录不建议提交到仓库。
- 假后端用 `time.sleep` 模拟阻塞调用，适合比较调度行为，不代表真实模型的耗时分布。真实 API 验证另做。
