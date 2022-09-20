from pathlib import Path

from napari_n2v.resources import *


def test_gear():
    assert Path(ICON_GEAR).exists()


def test_github():
    assert Path(ICON_GITHUB).exists()


def test_juglab():
    assert Path(ICON_JUGLAB).exists()


def test_tf():
    assert Path(ICON_TF).exists()


def test_bioimage():
    assert Path(DOC_BIOIMAGE).exists()
