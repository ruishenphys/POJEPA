"""交付入口的CPU检查；外设由明确虚构替身代替，不读取真实数据/权重。"""
from argparse import Namespace
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from po_jepa.runtime import cli, workflow as w


def test_help_does_not_load_assets_or_models(monkeypatch):
    """安装后help不能隐式启动计算或要求已有GPU。"""
    monkeypatch.setattr(w,'execute',lambda *a:pytest.fail('help must be inert'))
    with pytest.raises(SystemExit) as error:cli.main(['--help'])
    assert error.value.code==0


def test_production_entry_refuses_cpu_before_creating_output(monkeypatch,tmp_path):
    """虚构CPU测试与真实运行入口分开；没有随机小模型回退。"""
    monkeypatch.setattr(w,'verify_bundle',lambda:'fictional')
    args=Namespace(command='check',device='cpu',features_root=tmp_path/'no-data',output=tmp_path/'run')
    with pytest.raises(ValueError,match='cuda'):w.execute(args)
    assert not args.output.exists()


@pytest.mark.parametrize('stage',['preflight','training','evaluation'])
def test_workflow_failure_stops_and_preserves_stage(monkeypatch,tmp_path,stage):
    """只模拟调度依赖；失败时不执行后续阶段、不生成成功文件。"""
    events=[]
    monkeypatch.setattr(w,'verify_bundle',lambda:'fictional')
    monkeypatch.setattr(w,'require_calibrated_weights',lambda:None)
    def part(name,result=None):
        def call(*a,**k):
            events.append(name)
            if name==stage:raise RuntimeError('fictional '+stage)
            return result
        return call
    monkeypatch.setattr(w,'preflight',part('preflight',dict(hardware=dict(name='fictional'))))
    monkeypatch.setattr(w,'train_conditions',part('training'))
    monkeypatch.setattr(w,'evaluate_conditions',part('evaluation'))
    args=Namespace(command='run',device='cuda:0',features_root=tmp_path/'absent',output=tmp_path/'run')
    with pytest.raises(RuntimeError,match='fictional'):w.execute(args)
    assert events==['preflight','training','evaluation'][:['preflight','training','evaluation'].index(stage)+1]
    assert w.read(args.output/'FAILED.json')['stage']==stage
    assert not (args.output/'COMPLETE.json').exists()


def test_workflow_calls_all_phases_once_without_old_stage_records(monkeypatch,tmp_path):
    """统一run不要求历史A—F记录；仍先实际预检再训练、评价。"""
    events=[];monkeypatch.setattr(w,'verify_bundle',lambda:'fictional')
    monkeypatch.setattr(w,'require_calibrated_weights',lambda:None)
    monkeypatch.setattr(w,'preflight',lambda *a:(events.append('check') or dict(hardware={})))
    monkeypatch.setattr(w,'train_conditions',lambda *a:events.append('train'))
    monkeypatch.setattr(w,'evaluate_conditions',lambda *a:events.append('eval'))
    args=Namespace(command='run',device='cuda:0',features_root=tmp_path/'absent',output=tmp_path/'run')
    assert w.execute(args)==0 and events==['check','train','eval']
    assert (args.output/'COMPLETE.json').is_file()
    with pytest.raises(FileExistsError):w.execute(args)


def test_validation_verifier_uses_actual_producer_layout_and_raw_values(monkeypatch,tmp_path):
    """运行原验证producer写真实布局，再拒绝被改错的汇总均值，不改变容差。"""
    from test_real_training import unit_windows
    from test_training import e06_config
    c=w.comparison
    templates=unit_windows(c.data)
    def windows(root,rows,**kwargs):
        return templates
    monkeypatch.setattr(c.data,'load_windows',windows)
    folder=tmp_path/'validations/0000';folder.parent.mkdir()
    model=c.IdentityPredictor()
    metric=c.validate_condition(model,tmp_path/'absent',folder,torch.device('cpu'),e06_config())
    row=dict(step=0,**metric)
    report=dict(completed_updates=0,validations=[row],best=row)
    assert w.verify_selection(tmp_path,report)==row
    row['mean']+=1.
    with pytest.raises(ValueError,match='mean differs'):w.verify_selection(tmp_path,report)


def test_bundle_core_math_matches_source_map():
    """核对原始来源、当前字节与去文档语法树，允许注释清理而不改变执行逻辑。"""
    class WithoutDocstrings(ast.NodeTransformer):
        def visit_Module(self,node):
            self.generic_visit(node)
            if node.body and isinstance(node.body[0],ast.Expr):
                value=node.body[0].value
                if isinstance(value,ast.Constant) and isinstance(value.value,str):node.body.pop(0)
            return node
        visit_FunctionDef=visit_AsyncFunctionDef=visit_ClassDef=visit_Module
    source=w.read(w.ROOT/'SOURCE_MAP.json')
    checked=0
    for row in source['files']:
        path=w.ROOT/row['target']
        if 'candidate_sha256' in row:
            assert w.cache.file_sha(path)==row['candidate_sha256']
        if row['operation']=='identical_bytes' and not row['target'].endswith('/__init__.py'):
            assert w.cache.file_sha(path)==row['sha256']
        if 'executable_ast_sha256' in row:
            tree=WithoutDocstrings().visit(ast.parse(path.read_text(encoding='utf-8')))
            actual=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()
            assert actual==row['executable_ast_sha256']
            checked+=1
    assert checked==source['comment_cleanup']['python_modules_checked']==24
