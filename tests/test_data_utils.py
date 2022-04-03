import sys; sys.path.extend(['.', 'src'])

from src.training.dataset import remove_root


def test_remove_root_dir():
    assert remove_root('/a/b/c', 'a') == 'b/c'
    assert remove_root('/a/b/c', '/a') == 'b/c'
    assert remove_root('/ax/b/c', '/a') == '/ax/b/c'
    assert remove_root('ax/b/c', 'a') == 'ax/b/c'
    assert remove_root('ax/b/c', '/a') == 'ax/b/c'
