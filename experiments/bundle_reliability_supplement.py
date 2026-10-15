"""Package completed local supplement evidence, source and final optimizer states.

No PDFs, virtual environments, Git internals, credentials or external uploads.
Intermediate 50-epoch checkpoints remain in the original local work root.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'revision'))
from artifact_io import ROOT, SOURCE_ROOT
from run_reliability_supplement import load_manifest, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); load_manifest()
    output = args.output.resolve()
    if output.exists(): raise FileExistsError(f'Preserve existing delivery: {output}')
    summary = json.loads((ROOT/'reports/summary.json').read_text())
    verification = json.loads((ROOT/'reports/verification.json').read_text())
    if verification['status'] != 'passed' or summary['datasets'] != 10:
        raise ValueError('Evidence must be complete and verified before packaging')
    uniform = json.loads((ROOT/'evidence/uniform_ode_checks.json').read_text())
    if uniform['status'] != 'complete' or uniform['total'] != 640:
        raise ValueError('Independent uniform audit must cover every model and retain failures')
    prefix = 'EAAI-26-10676_reliability_supplement'
    files = {}
    for directory in ('src', 'revision', 'experiments', 'tests'):
        for path in sorted((SOURCE_ROOT/directory).rglob('*.py')):
            if '__pycache__' not in path.parts:
                files[f'{prefix}/code/{path.relative_to(SOURCE_ROOT)}'] = path
    for name in ('pyproject.toml', 'README.md', '.gitignore'):
        files[f'{prefix}/code/{name}'] = SOURCE_ROOT/name
    for directory in ('evidence', 'provenance', 'reports'):
        for path in sorted((ROOT/directory).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                if path.suffix not in ('.json', '.npz', '.xml', '.py', '.patch', '.md', '.png', '.csv', '.txt'):
                    raise ValueError(f'Unexpected evidence file type: {path}')
                files[f'{prefix}/results/{path.relative_to(ROOT)}'] = path
    for path in sorted((ROOT/'outputs/logs').glob('*.log')):
        files[f'{prefix}/results/{path.relative_to(ROOT)}'] = path
    for record in sorted((ROOT/'evidence/training').glob('*.json')):
        row = json.loads(record.read_text()); run = ROOT/'outputs/runs'/row['id']
        for path in sorted(run.glob('*.npz')):
            files[f'{prefix}/results/{path.relative_to(ROOT)}'] = path
        required = [run/'adam/history.json', run/'adam'/f"checkpoint_{row['final_adam_epoch']}.pt",
                    run/'lbfgs/history.json', run/'lbfgs/RESULT.json', run/'lbfgs/optimizer_checkpoint.pt']
        for path in required:
            if not path.is_file(): raise FileNotFoundError(path)
            files[f'{prefix}/results/{path.relative_to(ROOT)}'] = path
    for path in sorted((ROOT/'evidence/archive_inputs').glob('*.json')):
        files[f'{prefix}/reference/evidence/pysr_observation_refit/{path.name}'] = path
    performance = SOURCE_ROOT/'outputs/gpu-performance-2026-10-12'
    for name in ('README.md', 'RESULT.json', 'profile.txt'):
        files[f'{prefix}/gpu_performance/{name}'] = performance/name
    readme = '''# EAAI-26-10676 本地补实验增量资料包

先读 `results/reports/README.md`。它包含全部五种子统计、来源边界、训练预算、均匀/Fourier 动态与数值/状态有效性。

- `code/`：此次实际本地 Python 源码、测试和项目元数据；尚未推送的修改也包含在内。
- `results/`：10 份保存观测、协议、100 个新主对照模型、60 个历史 PySR 模型的审计、汇总 CSV/JSON/PNG、测试与环境快照。
- `results/outputs/runs/`：原生网络权重、闭合曲线、完整 Adam 历史以及每个输入的最终 Adam/L-BFGS 优化器断点。每 50 轮的其他断点仍保留在原本地目录，未重复打包。
- `reference/`：30 个已经完成的历史 PySR 重拟合 JSON，字节与引用记录相同；没有重新跑结构搜索。
- `gpu_performance/`：关于低功耗与 CPU/GPU 同步开销的隔离短测量，正式训练源代码未因此修改。

本包不包含虚拟环境、Git 私密配置、密钥或 PDF。没有改论文、Overleaf 原稿或现有理论草稿，也没有向任何 GitHub 仓库推送。此前 v3 完整资料包仍是论文与原审稿意见的独立来源；本包是新补实验的增量证据。

在已有兼容 Python 环境中，进入 `code/`，无需访问 sandbox 或运行训练即可重建报告：

```bash
<现有环境的python> experiments/report_reliability_supplement.py \\
  --work-root ../results --input-root ../results
```

首次解压可先核验 `MANIFEST.json` 中的文件 SHA256。报告脚本会再核验所有输入、原生教师导出、算法源签名、共同初值和闭合指标。源代码不可随意改动后继续把旧缓存称为同一实验。重新跑训练应使用保留的实际 Git 工作区和新的明确工作根；这是审阅/报告复建资料包，不是已经安装好的独立运行环境。

统计边界：新主对照是同输入的受限家族实验，模型空间和目标/先验不同；历史 PySR 仍保留旧教师身份。没有按真实误差选择预算、终点或最好种子，失败及越界模型均保留。有限预算或局部优化停止不构成全局恢复证书。
'''
    manifest = dict(generated_utc=datetime.now(timezone.utc).isoformat(),
        classification='Incremental completed reviewer supplement, local only',
        files=[dict(path=name.removeprefix(prefix+'/'), sha256=digest(path), bytes=path.stat().st_size)
               for name, path in sorted(files.items())],
        readme_sha256=hashlib.sha256(readme.encode()).hexdigest())
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.zip.partial')
    if temporary.exists(): raise FileExistsError(temporary)
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr(prefix+'/README.md', readme)
        archive.writestr(prefix+'/MANIFEST.json', json.dumps(manifest, indent=2))
        for name, path in sorted(files.items()): archive.write(path, name)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None: raise IOError('Bundle CRC check failed')
        for name, path in files.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest(path):
                raise IOError(f'Bundle file checksum failed: {name}')
    temporary.rename(output)
    print(json.dumps(dict(event='supplement_packaged', output=str(output), files=len(files)+2,
        bytes=output.stat().st_size, sha256=digest(output)), ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
