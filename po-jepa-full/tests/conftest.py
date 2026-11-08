"""仅控制CPU单元测试的线程；生产入口及随机流不由此配置。"""
import pytest
import torch


@pytest.fixture(scope='session',autouse=True)
def cpu_test_threads():
    """减少小型CPU算例线程争用，结束后恢复；不改种子或生产线程设置。"""
    before=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)
