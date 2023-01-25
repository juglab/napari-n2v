from pathlib import Path

from napari_n2v.resources import *


def test_gear():
    assert Path(ICON_GEAR).exists()
    assert type(ICON_GEAR) == str


def test_github():
    assert Path(ICON_GITHUB).exists()
    assert type(ICON_GITHUB) == str


def test_juglab():
    assert Path(ICON_JUGLAB).exists()
    assert type(ICON_JUGLAB) == str


def test_tf():
    assert Path(ICON_TF).exists()
    assert type(ICON_TF) == str
